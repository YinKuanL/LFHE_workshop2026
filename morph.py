from __future__ import annotations

import copy
import random
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Iterable, Mapping, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


StateDict = Dict[str, torch.Tensor]


class MorphNode:
    """
    Morph baseline aligned with the released bacox/Morph implementation.

    Reference behaviour:
    - fixed incoming sender view
    - layer-wise cosine model similarity
    - preference for dissimilar peers
    - gossip-based discovery of known nodes
    - indirect similarity estimation
    - one-add / one-remove topology updates
    - periodic topology adaptation
    - pull-style model aggregation

    Important
    ---------
    This class implements Morph's algorithmic node behaviour inside a
    synchronous simulator. The outer simulator is responsible for routing
    request and model messages between nodes.

    The simulator must NOT centrally choose Morph peers. Each node makes its
    own topology decision through `update_wanted_senders()`.
    """

    def __init__(
        self,
        node_id: int,
        model: nn.Module,
        init_senders: Iterable[int],
        *,
        initial_known_nodes: Optional[Iterable[int]] = None,
        in_degree: int = 4,
        beta: float = 500.0,
        change_iter: int = 5,
        seed: int = 42,
        indirect_history_k: int = 5,
        device: Optional[torch.device] = None,
    ) -> None:
        if in_degree <= 0:
            raise ValueError("in_degree must be positive.")

        if change_iter <= 0:
            raise ValueError("change_iter must be positive.")

        self.id = int(node_id)
        self.model = model

        self.device = (
            device
            if device is not None
            else next(model.parameters()).device
        )

        self.in_degree = int(in_degree)
        self.beta = float(beta)
        self.change_iter = int(change_iter)
        self.indirect_history_k = int(indirect_history_k)

        # Morph released implementation seeds node-local RNG separately.
        self.rng = random.Random(seed + self.id)

        initial = {
            int(peer)
            for peer in init_senders
            if int(peer) != self.id
        }

        if len(initial) != self.in_degree:
            raise ValueError(
                f"Node {self.id}: expected exactly {self.in_degree} "
                f"initial senders, got {len(initial)}."
            )

        # ------------------------------------------------------------
        # Incoming communication view:
        # peers whose models this node wants to receive.
        # ------------------------------------------------------------
        self.wanted_senders: Set[int] = set(initial)

        # ------------------------------------------------------------
        # Locally known node IDs.
        #
        # This can grow through gossip. Knowing a node does NOT mean
        # that its model is locally available.
        # ------------------------------------------------------------
        self.known_nodes: Set[int] = {
            int(peer) for peer in (initial_known_nodes or initial)
            if int(peer) != self.id
        }
        self.known_nodes.add(self.id)

        # ------------------------------------------------------------
        # Cached real peer models.
        #
        # peer_models is persistent across rounds and is used to compute
        # direct similarity to peers previously observed.
        # ------------------------------------------------------------
        self.peer_models: Dict[int, StateDict] = {}
        self.has_real_model: Set[int] = set()

        # ------------------------------------------------------------
        # Models actually requested / received in THIS round.
        #
        # Only these are eligible for aggregation.
        # ------------------------------------------------------------
        self.round_models: Dict[int, StateDict] = {}
        self.round_sender_degrees: Dict[int, int] = {}

        # Direct or estimated similarity cache:
        # higher = more similar.
        self.similarity_cache: Dict[int, float] = {}

        # target peer ->
        # deque[(iteration, intermediate_peer, sim(intermediate,target))]
        #
        # Used to estimate similarity to peers whose model has not been
        # directly received.
        self.sim_estimates_per_peer: Dict[
            int,
            Deque[Tuple[int, int, float]],
        ] = defaultdict(
            lambda: deque(maxlen=self.indirect_history_k)
        )

        self.iteration = 0

        # Diagnostic information.
        self.last_added_peer: Optional[int] = None
        self.last_removed_peer: Optional[int] = None
        self.topology_change_count = 0

    # ==============================================================
    # Model helpers
    # ==============================================================

    def _copy_state_dict(self, state: Mapping[str, torch.Tensor]) -> StateDict:
        return {
            name: tensor.detach().clone()
            for name, tensor in state.items()
        }

    def current_state_dict(self) -> StateDict:
        return self._copy_state_dict(self.model.state_dict())

    # ==============================================================
    # Morph model similarity
    # ==============================================================

    def cosine_similarity_state(
        self,
        peer_state: Mapping[str, torch.Tensor],
    ) -> float:
        """
        Compute Morph-style model similarity.

        Morph computes cosine similarity separately for compatible parameter
        tensors and averages the per-parameter similarities.

        This intentionally does NOT concatenate the entire model into one
        vector, because that would overweight large layers.
        """
        similarities = []

        with torch.no_grad():
            for name, my_param in self.model.named_parameters():
                if name not in peer_state:
                    continue

                peer_param = peer_state[name]

                my_tensor = (
                    my_param.detach()
                    .flatten()
                    .to(self.device)
                )

                peer_tensor = (
                    peer_param.detach()
                    .flatten()
                    .to(self.device)
                )

                if my_tensor.numel() != peer_tensor.numel():
                    continue

                my_norm = torch.linalg.vector_norm(my_tensor)
                peer_norm = torch.linalg.vector_norm(peer_tensor)

                if my_norm.item() == 0.0 or peer_norm.item() == 0.0:
                    continue

                similarity = F.cosine_similarity(
                    my_tensor,
                    peer_tensor,
                    dim=0,
                ).item()

                similarities.append(float(similarity))

        if not similarities:
            return -1.0

        return float(sum(similarities) / len(similarities))

    def compute_similarity(
        self,
        peer_id: int,
        fallback: float,
    ) -> float:
        """
        Direct similarity for peers whose real model has been observed.
        """
        if peer_id not in self.peer_models:
            return self.similarity_cache.get(peer_id, fallback)

        similarity = self.cosine_similarity_state(
            self.peer_models[peer_id]
        )

        self.similarity_cache[peer_id] = similarity

        return similarity

    # ==============================================================
    # Indirect similarity estimation
    # ==============================================================

    def estimate_similarity(
        self,
        peer_id: int,
        fallback: float,
    ) -> float:
        """
        Estimate similarity to a peer whose real model is unavailable.

        Released Morph/DissDL uses transitive estimates:

            sim(i, z) ~= sim(i, y) * sim(y, z)

        where y is an intermediate known peer.
        """
        reports = self.sim_estimates_per_peer.get(peer_id)

        if reports:
            estimates = []

            for _, intermediate_peer, sim_yz in reports:
                sim_iy = self.similarity_cache.get(
                    intermediate_peer
                )

                if sim_iy is None:
                    sim_iy = self.compute_similarity(
                        intermediate_peer,
                        fallback,
                    )

                estimates.append(
                    float(sim_iy) * float(sim_yz)
                )

            if estimates:
                estimate = float(
                    sum(estimates) / len(estimates)
                )

                self.similarity_cache[peer_id] = estimate
                return estimate

        if peer_id in self.similarity_cache:
            return self.similarity_cache[peer_id]

        return fallback

    # ==============================================================
    # Gossip payload
    # ==============================================================

    def known_similarity_payload(self) -> Dict[int, float]:
        """
        Similarity metadata advertised to other peers.

        Only locally available similarity estimates are shared.
        """
        return {
            peer_id: float(sim)
            for peer_id, sim in self.similarity_cache.items()
            if peer_id != self.id
            and sim != 0.0
        }

    def build_model_payload(self, degree: Optional[int] = None) -> Dict[str, Any]:
        """
        Construct the information sent when another node requests our model.

        The official implementation includes:
        - model parameters
        - known node IDs
        - known similarity estimates
        """
        return {
            "iteration": self.iteration,
            "params": self.current_state_dict(),
            "known_nodes": sorted(self.known_nodes),
            "known_similarities": self.known_similarity_payload(),
            "degree": int(degree) if degree is not None else len(self.wanted_senders),
        }

    def receive_model_payload(
        self,
        sender_id: int,
        payload: Mapping[str, Any],
    ) -> None:
        """
        Receive a model requested from a Morph sender.

        This performs three different operations:

        1. cache the real model;
        2. register it for this round's aggregation;
        3. process peer-discovery / similarity gossip metadata.
        """
        sender_id = int(sender_id)

        if sender_id == self.id:
            raise ValueError("A node cannot receive its own model.")

        payload_iteration = int(
            payload.get("iteration", self.iteration)
        )

        # Ignore stale model payloads.
        if payload_iteration < self.iteration:
            return

        params = payload.get("params")

        if params is None:
            raise ValueError(
                f"Payload from peer {sender_id} has no model parameters."
            )

        peer_state = self._copy_state_dict(params)

        first_time_seen = sender_id not in self.has_real_model

        self.peer_models[sender_id] = peer_state
        self.has_real_model.add(sender_id)

        # A direct model supersedes previously indirect-only information.
        if first_time_seen:
            self.sim_estimates_per_peer.pop(
                sender_id,
                None,
            )

        # Only requested senders should participate in aggregation.
        if sender_id in self.wanted_senders:
            self.round_models[sender_id] = peer_state
            self.round_sender_degrees[sender_id] = int(payload.get("degree", len(self.wanted_senders)))

        self.known_nodes.add(sender_id)

        for peer in payload.get("known_nodes", []):
            peer = int(peer)

            if peer != self.id:
                self.known_nodes.add(peer)

        # ----------------------------------------------------------
        # Morph collects similarity gossip shortly before topology
        # adaptation.
        #
        # Official released code records these estimates when:
        #
        # iteration % (change_iter - 1) == 0
        #
        # for change_iter > 1.
        # ----------------------------------------------------------
        should_collect_gossip = (
            self.iteration > 0
            and self.change_iter > 1
            and self.iteration % (self.change_iter - 1) == 0
        )

        # For change_iter=1 there is no preceding discovery round,
        # therefore accept gossip every round.
        if self.change_iter == 1:
            should_collect_gossip = True

        if should_collect_gossip:
            similarity_map = payload.get(
                "known_similarities",
                {},
            )

            for target_peer, similarity in similarity_map.items():
                target_peer = int(target_peer)

                if target_peer == self.id:
                    continue

                # No indirect estimate is needed once we possess
                # the actual peer model.
                if target_peer in self.has_real_model:
                    continue

                self.known_nodes.add(target_peer)

                self.sim_estimates_per_peer[target_peer].append(
                    (
                        self.iteration,
                        sender_id,
                        float(similarity),
                    )
                )

    # ==============================================================
    # Morph topology adaptation
    # ==============================================================

    def _fallback_similarity(self) -> float:
        if not self.similarity_cache:
            return 0.0

        return float(
            sum(self.similarity_cache.values())
            / len(self.similarity_cache)
        )

    def _sample_softmax(
        self,
        scores: Mapping[int, float],
    ) -> int:
        """
        Sample exactly one peer according to softmax(beta * score).
        """
        if not scores:
            raise ValueError("Cannot sample from an empty score dictionary.")

        peer_ids = list(scores.keys())

        score_tensor = torch.tensor(
            [scores[p] for p in peer_ids],
            # Pinned Morph/DissDL uses float32 before softmax. At beta=500,
            # changing this dtype can change seeded categorical selections.
            dtype=torch.float32,
        )

        probs = torch.softmax(
            self.beta * score_tensor,
            dim=0,
        ).tolist()

        return int(
            self.rng.choices(
                peer_ids,
                weights=probs,
                k=1,
            )[0]
        )

    def update_wanted_senders(
        self,
        iteration: int,
        *,
        available_peers: Optional[Iterable[int]] = None,
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Perform the released Morph one-add / one-remove topology update.

        Returns
        -------
        (peer_added, peer_removed)

        If no update takes place:
            (None, None)
        """
        self.iteration = int(iteration)

        self.last_added_peer = None
        self.last_removed_peer = None

        if self.iteration == 0:
            return None, None

        if self.iteration % self.change_iter != 0:
            return None, None

        current_senders = set(self.wanted_senders)

        if available_peers is None:
            possible_peers = set(self.known_nodes)
        else:
            possible_peers = {
                int(p)
                for p in available_peers
            }

            # A peer must still have been discovered locally.
            possible_peers &= self.known_nodes

        possible_peers.discard(self.id)

        add_candidates = list(
            possible_peers - current_senders
        )

        remove_candidates = list(current_senders)

        # Released implementation keeps at least one sender.
        if len(remove_candidates) <= 1:
            return None, None

        if not add_candidates:
            return None, None

        fallback = self._fallback_similarity()

        # ----------------------------------------------------------
        # ADD:
        # low similarity => high desirability
        #
        # score = -similarity
        # ----------------------------------------------------------
        add_scores: Dict[int, float] = {}

        for peer_id in add_candidates:
            if peer_id in self.has_real_model:
                similarity = self.compute_similarity(
                    peer_id,
                    fallback,
                )
            else:
                similarity = self.estimate_similarity(
                    peer_id,
                    fallback,
                )

            add_scores[peer_id] = -float(similarity)

        peer_to_add = self._sample_softmax(
            add_scores
        )

        # ----------------------------------------------------------
        # REMOVE:
        # high similarity => high removal probability
        #
        # score = +similarity
        # ----------------------------------------------------------
        remove_scores: Dict[int, float] = {}

        for peer_id in remove_candidates:
            if peer_id in self.has_real_model:
                similarity = self.compute_similarity(
                    peer_id,
                    fallback,
                )
            else:
                similarity = self.estimate_similarity(
                    peer_id,
                    fallback,
                )

            remove_scores[peer_id] = float(similarity)

        peer_to_remove = self._sample_softmax(
            remove_scores
        )

        candidate_senders = set(current_senders)

        candidate_senders.add(peer_to_add)
        candidate_senders.discard(peer_to_remove)

        # Morph maintains a fixed incoming view.
        if len(candidate_senders) != self.in_degree:
            raise RuntimeError(
                f"Node {self.id}: topology update broke fixed "
                f"in-degree constraint: "
                f"{len(candidate_senders)} != {self.in_degree}"
            )

        self.wanted_senders = candidate_senders

        self.last_added_peer = peer_to_add
        self.last_removed_peer = peer_to_remove
        self.topology_change_count += 1

        # Dropped sender should not contribute an old model to the
        # next aggregation.
        self.round_models.pop(
            peer_to_remove,
            None,
        )

        return peer_to_add, peer_to_remove

    # ==============================================================
    # Pull / request semantics
    # ==============================================================

    def requested_senders(self) -> Set[int]:
        """
        Peers from whom this node requests a model this round.
        """
        return set(self.wanted_senders)

    def should_send_to(
        self,
        requester_id: int,
        requester_wants_model: bool,
    ) -> bool:
        """
        Node-local response to a simulated DPSGD_REQ.

        The Morph release sends a model only to peers that request it.
        """
        requester_id = int(requester_id)

        if requester_id == self.id:
            return False

        return bool(requester_wants_model)

    # ==============================================================
    # Aggregation
    # ==============================================================

    def aggregate(self) -> int:
        """
        Uniformly average:

            self model + models received from wanted_senders.

        Returns
        -------
        int
            Number of peer models aggregated.

        IMPORTANT:
        Discovery metadata and cached candidate models do NOT
        automatically participate in aggregation.
        """
        valid_models = {
            peer_id: state
            for peer_id, state in self.round_models.items()
            if peer_id in self.wanted_senders
        }

        if not valid_models:
            self.round_models.clear()
            self.round_sender_degrees.clear()
            return 0

        averaged_state: StateDict = {}

        with torch.no_grad():
            local_state = self.current_state_dict()
            peer_count = len(valid_models)
            weights: Dict[int, float] = {}
            for peer_id in valid_models:
                sender_degree = self.round_sender_degrees.get(peer_id, peer_count)
                weights[peer_id] = 1.0 / (max(peer_count, sender_degree) + 1)
            weight_total = sum(weights.values())
            for key, local_value in local_state.items():
                total = None
                for peer_id, peer_state in valid_models.items():
                    term = peer_state[key].to(self.device) * weights[peer_id]
                    total = term if total is None else total + term
                self_term = local_value.to(self.device) * (1.0 - weight_total)
                averaged_state[key] = self_term if total is None else total + self_term

            self.model.load_state_dict(
                averaged_state,
                strict=True,
            )

        num_aggregated = len(valid_models)

        self.round_models.clear()
        self.round_sender_degrees.clear()

        return num_aggregated

    # ==============================================================
    # Round control / diagnostics
    # ==============================================================

    def begin_round(self, iteration: int) -> None:
        self.iteration = int(iteration)
        self.round_models.clear()
        self.round_sender_degrees.clear()

    def validate_state(self) -> None:
        if self.id in self.wanted_senders:
            raise RuntimeError(
                f"Node {self.id} has itself as a sender."
            )

        if len(self.wanted_senders) != self.in_degree:
            raise RuntimeError(
                f"Node {self.id}: expected fixed in-degree "
                f"{self.in_degree}, got "
                f"{len(self.wanted_senders)}."
            )

        if len(set(self.wanted_senders)) != len(
            self.wanted_senders
        ):
            raise RuntimeError(
                f"Node {self.id}: duplicate sender."
            )

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "node_id": self.id,
            "iteration": self.iteration,
            "in_degree": len(self.wanted_senders),
            "wanted_senders": sorted(self.wanted_senders),
            "known_nodes": len(self.known_nodes - {self.id}),
            "real_models_cached": len(self.has_real_model),
            "similarities_cached": len(self.similarity_cache),
            "indirect_targets": len(self.sim_estimates_per_peer),
            "last_added_peer": self.last_added_peer,
            "last_removed_peer": self.last_removed_peer,
            "topology_change_count": self.topology_change_count,
        }
