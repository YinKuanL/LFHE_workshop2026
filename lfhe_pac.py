from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

import networkx as nx
import numpy as np


Edge = tuple[int, int]


class LFHEPACInvariantError(RuntimeError):
    """Raised when a PAC transaction would violate graph/protocol state."""


def canonical_edge(edge: Edge) -> Edge:
    left, right = map(int, edge)
    return (left, right) if left < right else (right, left)


def _hash_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest().upper()


def _edge_payload(edges: Iterable[Edge]) -> list[list[int]]:
    return [list(edge) for edge in sorted(canonical_edge(edge) for edge in edges)]


@dataclass(frozen=True)
class LFHEPACSnapshot:
    num_nodes: int
    protected_edges: frozenset[Edge]
    adaptive_edges: frozenset[Edge]
    dmax: int
    edge_budget: int
    versions: tuple[tuple[int, int], ...]
    representations: tuple[tuple[int, np.ndarray], ...]
    representation_timestamp: int

    @property
    def graph(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_nodes))
        graph.add_edges_from(self.protected_edges | self.adaptive_edges)
        return graph

    def version_for(self, endpoint: int) -> int:
        return dict(self.versions)[int(endpoint)]

    def representation_for(self, endpoint: int) -> np.ndarray:
        return dict(self.representations)[int(endpoint)]

    @property
    def topology_hash(self) -> str:
        return _hash_json(
            {
                "nodes": self.num_nodes,
                "protected": _edge_payload(self.protected_edges),
                "adaptive": _edge_payload(self.adaptive_edges),
                "dmax": self.dmax,
                "edge_budget": self.edge_budget,
            }
        )

    @property
    def protected_hash(self) -> str:
        return _hash_json(_edge_payload(self.protected_edges))

    @property
    def representation_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.representation_timestamp).encode())
        for endpoint, representation in self.representations:
            digest.update(str(endpoint).encode())
            digest.update(np.asarray(representation, dtype=np.float64).tobytes())
        return digest.hexdigest().upper()


class LFHEPACState:
    """Official topology plus endpoint-owned versions and transient locks."""

    def __init__(
        self,
        *,
        num_nodes: int,
        protected_edges: Iterable[Edge],
        adaptive_edges: Iterable[Edge],
        dmax: int,
        edge_budget: int,
    ) -> None:
        self.num_nodes = int(num_nodes)
        self.protected_edges = frozenset(canonical_edge(e) for e in protected_edges)
        self.adaptive_edges = frozenset(canonical_edge(e) for e in adaptive_edges)
        self.dmax = int(dmax)
        self.edge_budget = int(edge_budget)
        self.versions = {node: 0 for node in range(self.num_nodes)}
        self.locks: dict[int, str] = {}
        self.validate()

    @property
    def graph(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_nodes))
        graph.add_edges_from(self.protected_edges | self.adaptive_edges)
        return graph

    @property
    def edge_count(self) -> int:
        return len(self.protected_edges) + len(self.adaptive_edges)

    @property
    def topology_hash(self) -> str:
        return _hash_json(
            {
                "nodes": self.num_nodes,
                "protected": _edge_payload(self.protected_edges),
                "adaptive": _edge_payload(self.adaptive_edges),
                "dmax": self.dmax,
                "edge_budget": self.edge_budget,
            }
        )

    @property
    def protected_hash(self) -> str:
        return _hash_json(_edge_payload(self.protected_edges))

    def clone(self) -> "LFHEPACState":
        clone = LFHEPACState(
            num_nodes=self.num_nodes,
            protected_edges=self.protected_edges,
            adaptive_edges=self.adaptive_edges,
            dmax=self.dmax,
            edge_budget=self.edge_budget,
        )
        clone.versions = dict(self.versions)
        clone.locks = dict(self.locks)
        return clone

    def fingerprint(self) -> str:
        return _hash_json(
            {
                "topology": self.topology_hash,
                "versions": sorted(self.versions.items()),
                "locks": sorted(self.locks.items()),
            }
        )

    def checkpoint(self) -> dict[str, object]:
        """JSON/torch-serializable official protocol state (locks are transient)."""
        return {
            "num_nodes": self.num_nodes,
            "protected_edges": _edge_payload(self.protected_edges),
            "adaptive_edges": _edge_payload(self.adaptive_edges),
            "dmax": self.dmax,
            "edge_budget": self.edge_budget,
            "versions": dict(self.versions),
            "locks": dict(self.locks),
        }

    @classmethod
    def restore(cls, value: Mapping[str, object]) -> "LFHEPACState":
        state = cls(
            num_nodes=int(value["num_nodes"]),
            protected_edges=value["protected_edges"],
            adaptive_edges=value["adaptive_edges"],
            dmax=int(value["dmax"]),
            edge_budget=int(value["edge_budget"]),
        )
        state.versions = {int(k): int(v) for k, v in dict(value["versions"]).items()}
        state.locks = {int(k): str(v) for k, v in dict(value.get("locks", {})).items()}
        state.validate()
        return state

    def validate(self) -> None:
        all_edges = self.protected_edges | self.adaptive_edges
        if self.protected_edges & self.adaptive_edges:
            raise LFHEPACInvariantError("protected/adaptive edge overlap")
        if any(left == right for left, right in all_edges):
            raise LFHEPACInvariantError("self-loop")
        graph = self.graph
        if not nx.is_connected(graph):
            raise LFHEPACInvariantError("graph disconnected")
        protected = nx.Graph()
        protected.add_nodes_from(range(self.num_nodes))
        protected.add_edges_from(self.protected_edges)
        if len(self.protected_edges) != self.num_nodes - 1 or not nx.is_tree(protected):
            raise LFHEPACInvariantError("protected edges are not a spanning tree")
        if max(dict(graph.degree()).values(), default=0) > self.dmax:
            raise LFHEPACInvariantError("hard Dmax exceeded")
        if graph.number_of_edges() > self.edge_budget:
            raise LFHEPACInvariantError("edge budget exceeded")

    def snapshot(
        self,
        representations: Mapping[int, np.ndarray],
        *,
        timestamp: int,
    ) -> LFHEPACSnapshot:
        if set(representations) != set(range(self.num_nodes)):
            raise ValueError("one frozen representation is required per endpoint")
        frozen = tuple(
            (
                node,
                np.array(representations[node], dtype=np.float64, copy=True),
            )
            for node in range(self.num_nodes)
        )
        for _, value in frozen:
            value.setflags(write=False)
        return LFHEPACSnapshot(
            num_nodes=self.num_nodes,
            protected_edges=self.protected_edges,
            adaptive_edges=self.adaptive_edges,
            dmax=self.dmax,
            edge_budget=self.edge_budget,
            versions=tuple(sorted(self.versions.items())),
            representations=frozen,
            representation_timestamp=int(timestamp),
        )


def _deterministic_spanning_tree(graph: nx.Graph, seed: int) -> frozenset[Edge]:
    weighted = nx.Graph()
    weighted.add_nodes_from(graph.nodes())
    edges = sorted(canonical_edge(edge) for edge in graph.edges())
    order = list(range(len(edges)))
    random.Random(seed * 1_000_003 + 17).shuffle(order)
    for rank, edge_index in enumerate(order):
        weighted.add_edge(*edges[edge_index], weight=rank)
    tree = nx.minimum_spanning_tree(weighted, algorithm="kruskal", weight="weight")
    return frozenset(canonical_edge(edge) for edge in tree.edges())


def build_random_heterogeneous_state(
    *,
    num_nodes: int,
    average_degree: int,
    dmax: int,
    seed: int,
    edge_budget: int | None = None,
    max_attempts: int = 100_000,
) -> LFHEPACState:
    """Construct a deterministic connected, heterogeneous hard-capped graph."""

    target_edges = int(round(num_nodes * average_degree / 2.0))
    capacity = num_nodes * dmax // 2
    if target_edges < num_nodes - 1 or target_edges > capacity:
        raise ValueError("requested average degree is incompatible with connectivity/Dmax")
    # A shuffled Hamiltonian cycle provides connectivity without rejection.
    # Deterministically shuffled chords then fill the exact degree-3 budget.
    # This avoids the vanishing acceptance rate of capped G(n,m) at N=500.
    for attempt in range(min(max_attempts, 256)):
        rng = random.Random(seed * 1_000_003 + attempt * 104_729)
        order = list(range(num_nodes)); rng.shuffle(order)
        graph = nx.Graph(); graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from((order[i], order[(i + 1) % num_nodes]) for i in range(num_nodes))
        pairs = [(u, v) for u in range(num_nodes) for v in range(u + 1, num_nodes)]
        rng.shuffle(pairs)
        for u, v in pairs:
            if graph.number_of_edges() == target_edges:
                break
            if not graph.has_edge(u, v) and graph.degree(u) < dmax and graph.degree(v) < dmax:
                graph.add_edge(u, v)
        degrees = dict(graph.degree())
        if graph.number_of_edges() != target_edges or len(set(degrees.values())) < 2:
            continue
        protected = _deterministic_spanning_tree(graph, seed)
        adaptive = frozenset(canonical_edge(edge) for edge in graph.edges()) - protected
        return LFHEPACState(
            num_nodes=num_nodes,
            protected_edges=protected,
            adaptive_edges=adaptive,
            dmax=dmax,
            edge_budget=capacity if edge_budget is None else int(edge_budget),
        )
    raise RuntimeError("could not generate a capped connected heterogeneous graph")


def normalized_structural_score(
    endpoint: int,
    graph: nx.Graph,
    representations: Mapping[int, np.ndarray],
) -> float:
    """Mean squared representation distance, accumulated in float64."""

    neighbors = sorted(int(peer) for peer in graph.neighbors(int(endpoint)))
    if not neighbors:
        return 0.0
    own = np.asarray(representations[int(endpoint)], dtype=np.float64)
    total = np.float64(0.0)
    for peer in neighbors:
        gap = own - np.asarray(representations[peer], dtype=np.float64)
        total += np.sum(gap * gap, dtype=np.float64)
    return float(total / np.float64(len(neighbors)))


@dataclass(frozen=True)
class FoFCandidatePacket:
    epoch: int
    slot: int
    initiator: int
    intermediate: int
    candidate: int

    @property
    def path(self) -> tuple[int, int, int]:
        return (self.initiator, self.intermediate, self.candidate)


@dataclass(frozen=True)
class FrozenFoFStream:
    epoch: int
    topology_hash: str
    representation_hash: str
    candidate_budget: int
    packets: tuple[FoFCandidatePacket, ...]

    @property
    def stream_hash(self) -> str:
        return _hash_json(
            {
                "epoch": self.epoch,
                "topology": self.topology_hash,
                "representation": self.representation_hash,
                "candidate_budget": self.candidate_budget,
                "packets": [packet.__dict__ for packet in self.packets],
            }
        )


def discover_frozen_fof(
    snapshot: LFHEPACSnapshot,
    *,
    candidate_budget: int,
    seed: int,
    initiator_order: Sequence[int] | None = None,
) -> FrozenFoFStream:
    if candidate_budget <= 0:
        raise ValueError("candidate_budget must be positive")
    graph = snapshot.graph
    order = list(range(snapshot.num_nodes)) if initiator_order is None else list(initiator_order)
    if set(order) != set(range(snapshot.num_nodes)):
        raise ValueError("initiator_order must contain every endpoint exactly once")
    packets: list[FoFCandidatePacket] = []
    for initiator in order:
        # Canonicalize each candidate to one deterministic FoF path before sampling.
        path_by_candidate: dict[int, tuple[int, int, int]] = {}
        for intermediate in sorted(graph.neighbors(initiator)):
            for candidate in sorted(graph.neighbors(intermediate)):
                candidate = int(candidate)
                if candidate == initiator or graph.has_edge(initiator, candidate):
                    continue
                path_by_candidate.setdefault(
                    candidate, (int(initiator), int(intermediate), candidate)
                )
        paths = list(path_by_candidate.values())
        random.Random(
            seed * 10_000_019
            + snapshot.representation_timestamp * 1_000_003
            + int(initiator) * 9_973
        ).shuffle(paths)
        for slot, path in enumerate(paths[:candidate_budget]):
            packets.append(
                FoFCandidatePacket(
                    epoch=snapshot.representation_timestamp,
                    slot=slot,
                    initiator=path[0],
                    intermediate=path[1],
                    candidate=path[2],
                )
            )
    return FrozenFoFStream(
        epoch=snapshot.representation_timestamp,
        topology_hash=snapshot.topology_hash,
        representation_hash=snapshot.representation_hash,
        candidate_budget=candidate_budget,
        packets=tuple(sorted(packets, key=lambda p: (p.initiator, p.slot, p.path))),
    )


@dataclass(frozen=True)
class LFHEPACProposal:
    txid: str
    operation: str
    initiator: int
    fof_path: tuple[int, int, int]
    removed_neighbor: int | None
    old_edge: Edge | None
    new_edge: Edge
    affected_endpoints: tuple[int, ...]
    expected_versions: tuple[tuple[int, int], ...]
    representation_timestamp: int
    scores_before: tuple[tuple[int, float], ...]
    scores_after: tuple[tuple[int, float], ...]
    endpoint_gains: tuple[tuple[int, float], ...]
    evidence_hash: str

    def gain_for(self, endpoint: int) -> float:
        return dict(self.endpoint_gains)[int(endpoint)]

    @property
    def initiator_gain(self) -> float:
        return self.gain_for(self.initiator)

    @property
    def minimum_gain(self) -> float:
        return min(gain for _, gain in self.endpoint_gains)


def _proposal_evidence_payload(proposal: LFHEPACProposal) -> dict[str, object]:
    return {
        "txid": proposal.txid,
        "operation": proposal.operation,
        "initiator": proposal.initiator,
        "path": proposal.fof_path,
        "removed": proposal.removed_neighbor,
        "old_edge": proposal.old_edge,
        "new_edge": proposal.new_edge,
        "affected": proposal.affected_endpoints,
        "versions": proposal.expected_versions,
        "timestamp": proposal.representation_timestamp,
        "before": proposal.scores_before,
        "after": proposal.scores_after,
        "gains": proposal.endpoint_gains,
    }


def _make_proposal(
    snapshot: LFHEPACSnapshot,
    packet: FoFCandidatePacket,
    *,
    removed_neighbor: int | None,
) -> LFHEPACProposal:
    operation = "addition" if removed_neighbor is None else "swap"
    old_edge = (
        None
        if removed_neighbor is None
        else canonical_edge((packet.initiator, removed_neighbor))
    )
    new_edge = canonical_edge((packet.initiator, packet.candidate))
    affected = tuple(
        sorted(
            {packet.initiator, packet.candidate}
            | ({int(removed_neighbor)} if removed_neighbor is not None else set())
        )
    )
    candidate_graph = snapshot.graph.copy()
    if old_edge is not None:
        candidate_graph.remove_edge(*old_edge)
    candidate_graph.add_edge(*new_edge)
    representations = dict(snapshot.representations)
    before = tuple(
        (q, normalized_structural_score(q, snapshot.graph, representations))
        for q in affected
    )
    after = tuple(
        (q, normalized_structural_score(q, candidate_graph, representations))
        for q in affected
    )
    gains = tuple(
        (q, dict(after)[q] - dict(before)[q])
        for q in affected
    )
    old_text = "none" if old_edge is None else f"{old_edge[0]}_{old_edge[1]}"
    txid = (
        f"e{snapshot.representation_timestamp}-{operation[0]}-i{packet.initiator}"
        f"-p{packet.intermediate}_{packet.candidate}-r{old_text}"
    )
    provisional = LFHEPACProposal(
        txid=txid,
        operation=operation,
        initiator=packet.initiator,
        fof_path=packet.path,
        removed_neighbor=removed_neighbor,
        old_edge=old_edge,
        new_edge=new_edge,
        affected_endpoints=affected,
        expected_versions=tuple((q, snapshot.version_for(q)) for q in affected),
        representation_timestamp=snapshot.representation_timestamp,
        scores_before=before,
        scores_after=after,
        endpoint_gains=gains,
        evidence_hash="",
    )
    return replace(provisional, evidence_hash=_hash_json(_proposal_evidence_payload(provisional)))


def enumerate_feasible_operations(
    snapshot: LFHEPACSnapshot,
    stream: FrozenFoFStream,
    *,
    initiator_order: Sequence[int] | None = None,
) -> tuple[LFHEPACProposal, ...]:
    if stream.topology_hash != snapshot.topology_hash:
        raise ValueError("candidate stream topology mismatch")
    if stream.representation_hash != snapshot.representation_hash:
        raise ValueError("candidate stream representation mismatch")
    graph = snapshot.graph
    order = list(range(snapshot.num_nodes)) if initiator_order is None else list(initiator_order)
    rank = {node: index for index, node in enumerate(order)}
    if set(rank) != set(range(snapshot.num_nodes)):
        raise ValueError("initiator_order must contain every endpoint exactly once")
    proposals: dict[tuple[object, ...], LFHEPACProposal] = {}
    for packet in sorted(stream.packets, key=lambda p: (rank[p.initiator], p.slot, p.path)):
        i, k = packet.initiator, packet.candidate
        if i == k or graph.has_edge(i, k):
            continue
        if (
            graph.degree(i) < snapshot.dmax
            and graph.degree(k) < snapshot.dmax
            and graph.number_of_edges() < snapshot.edge_budget
        ):
            proposal = _make_proposal(snapshot, packet, removed_neighbor=None)
            proposals[
                (
                    proposal.initiator,
                    proposal.operation,
                    proposal.new_edge,
                    proposal.old_edge,
                )
            ] = proposal
        if graph.degree(k) >= snapshot.dmax:
            continue
        for removed in sorted(graph.neighbors(i)):
            old_edge = canonical_edge((i, int(removed)))
            if old_edge not in snapshot.adaptive_edges:
                continue
            if graph.degree(removed) - 1 < 1:
                continue
            proposal = _make_proposal(snapshot, packet, removed_neighbor=int(removed))
            proposals[
                (
                    proposal.initiator,
                    proposal.operation,
                    proposal.new_edge,
                    proposal.old_edge,
                )
            ] = proposal
    return tuple(sorted(proposals.values(), key=lambda proposal: proposal.txid))


def feasible_operation_hash(proposals: Sequence[LFHEPACProposal]) -> str:
    return _hash_json(
        [
            {
                "txid": proposal.txid,
                "operation": proposal.operation,
                "old": proposal.old_edge,
                "new": proposal.new_edge,
                "affected": proposal.affected_endpoints,
            }
            for proposal in sorted(proposals, key=lambda item: item.txid)
        ]
    )


def _pac_priority(proposal: LFHEPACProposal) -> tuple[float, float, int, str]:
    return (
        -proposal.initiator_gain,
        -proposal.minimum_gain,
        0 if proposal.operation == "addition" else 1,
        proposal.txid,
    )


def _passes_method_score(proposal: LFHEPACProposal, method: str) -> bool:
    if method == "random_fof_matched":
        return True
    if proposal.initiator_gain <= 0.0:
        return False
    if method in {"lfhe_pac_initiator_only", "lfhe_pac"}:
        return True
    raise ValueError(f"unsupported PAC method: {method}")


def select_one_proposal_per_initiator(
    feasible: Sequence[LFHEPACProposal],
    *,
    method: str,
    seed: int,
) -> tuple[LFHEPACProposal, ...]:
    selected: list[LFHEPACProposal] = []
    initiators = sorted({proposal.initiator for proposal in feasible})
    for initiator in initiators:
        options = [
            proposal
            for proposal in feasible
            if proposal.initiator == initiator and _passes_method_score(proposal, method)
        ]
        if not options:
            continue
        if method == "random_fof_matched":
            options = sorted(options, key=lambda proposal: proposal.txid)
            choice = random.Random(
                seed * 1_000_003
                + options[0].representation_timestamp * 97_409
                + initiator * 9_973
            ).choice(options)
        else:
            choice = min(options, key=_pac_priority)
        selected.append(choice)
    return tuple(sorted(selected, key=lambda proposal: proposal.txid))


@dataclass(frozen=True)
class EndpointResponse:
    txid: str
    endpoint_id: int
    granted: bool
    reason: str
    topology_version: int
    evidence_hash: str


@dataclass(frozen=True)
class TransactionOutcome:
    txid: str
    committed: bool
    reason: str


@dataclass(frozen=True)
class PACEpochResult:
    outcomes: tuple[TransactionOutcome, ...]
    responses: tuple[EndpointResponse, ...]
    committed_additions: int
    committed_swaps: int
    conflict_rejections: int
    stale_rejections: int
    timeout_aborts: int
    budget_rejections: int
    endpoint_vetoes: tuple[tuple[str, int], ...]
    control_messages: int
    control_bytes: int

    @property
    def committed_transactions(self) -> int:
        return self.committed_additions + self.committed_swaps


def _recompute_evidence(proposal: LFHEPACProposal) -> str:
    without_hash = replace(proposal, evidence_hash="")
    return _hash_json(_proposal_evidence_payload(without_hash))


def _topology_validation_reason(
    state: LFHEPACState,
    snapshot: LFHEPACSnapshot,
    proposal: LFHEPACProposal,
    endpoint: int,
) -> str | None:
    if endpoint not in proposal.affected_endpoints:
        return "endpoint_not_affected"
    if proposal.representation_timestamp != snapshot.representation_timestamp:
        return "stale_representation_timestamp"
    if proposal.evidence_hash != _recompute_evidence(proposal):
        return "inconsistent_evidence"
    expected = dict(proposal.expected_versions)[endpoint]
    if state.versions[endpoint] != expected:
        return "stale_topology_version"
    if endpoint in state.locks:
        return "endpoint_locked"
    graph = state.graph
    if proposal.old_edge is not None and endpoint in proposal.old_edge:
        if proposal.old_edge not in state.adaptive_edges:
            return "old_edge_missing_or_protected"
    if endpoint in proposal.new_edge and graph.has_edge(*proposal.new_edge):
        return "new_edge_exists"
    candidate = graph.copy()
    if proposal.old_edge is not None:
        if not candidate.has_edge(*proposal.old_edge):
            return "old_edge_missing_or_protected"
        candidate.remove_edge(*proposal.old_edge)
    candidate.add_edge(*proposal.new_edge)
    if candidate.degree(endpoint) > state.dmax:
        return "hard_degree_cap"
    if proposal.operation == "swap" and proposal.removed_neighbor is not None:
        if candidate.degree(proposal.removed_neighbor) < 1:
            return "removed_endpoint_isolated"
    if candidate.number_of_edges() > state.edge_budget:
        return "edge_budget"
    return None


def _score_validation_reason(
    snapshot: LFHEPACSnapshot,
    proposal: LFHEPACProposal,
    endpoint: int,
    method: str,
) -> str | None:
    if method == "random_fof_matched":
        return None
    old_graph = snapshot.graph
    proposed_graph = old_graph.copy()
    if proposal.old_edge is not None:
        proposed_graph.remove_edge(*proposal.old_edge)
    proposed_graph.add_edge(*proposal.new_edge)
    representations = dict(snapshot.representations)
    gain = normalized_structural_score(
        endpoint, proposed_graph, representations
    ) - normalized_structural_score(endpoint, old_graph, representations)
    if gain != proposal.gain_for(endpoint):
        return "inconsistent_local_score_evidence"
    if endpoint == proposal.initiator:
        return None if gain > 0.0 else "initiator_score_veto"
    if method == "lfhe_pac_initiator_only":
        return None
    if method == "lfhe_pac":
        return None if gain >= 0.0 else "noninitiator_score_veto"
    raise ValueError(f"unsupported PAC method: {method}")


def _endpoint_priority(
    proposal: LFHEPACProposal,
    *,
    method: str,
    seed: int,
) -> tuple[object, ...]:
    if method == "random_fof_matched":
        digest = hashlib.sha256(f"{seed}:{proposal.txid}".encode()).hexdigest()
        return (digest, proposal.txid)
    return _pac_priority(proposal)


def _atomic_apply(state: LFHEPACState, proposal: LFHEPACProposal) -> bool:
    before = state.fingerprint()
    candidate = state.clone()
    try:
        if any(
            candidate.versions[endpoint] != expected
            for endpoint, expected in proposal.expected_versions
        ):
            raise LFHEPACInvariantError("stale topology version")
        adaptive = set(candidate.adaptive_edges)
        if proposal.old_edge is not None:
            if proposal.old_edge not in adaptive:
                raise LFHEPACInvariantError("old edge missing or protected")
            adaptive.remove(proposal.old_edge)
        if proposal.new_edge in candidate.protected_edges | frozenset(adaptive):
            raise LFHEPACInvariantError("new edge already exists")
        adaptive.add(proposal.new_edge)
        candidate.adaptive_edges = frozenset(adaptive)
        for endpoint in proposal.affected_endpoints:
            candidate.versions[endpoint] += 1
        candidate.locks = {}
        candidate.validate()
    except Exception:
        if state.fingerprint() != before:
            raise AssertionError("failed commit mutated official state")
        return False
    state.adaptive_edges = candidate.adaptive_edges
    state.versions = candidate.versions
    state.locks = {}
    state.validate()
    return True


def run_pac_epoch(
    state: LFHEPACState,
    snapshot: LFHEPACSnapshot,
    proposals: Sequence[LFHEPACProposal],
    *,
    method: str,
    max_commits: int,
    seed: int,
    force_timeout_txids: Iterable[str] = (),
) -> PACEpochResult:
    """Simulate endpoint-local arbitration and atomic PAC transactions."""

    if method not in {"random_fof_matched", "lfhe_pac_initiator_only", "lfhe_pac"}:
        raise ValueError(f"unsupported PAC method: {method}")
    timeouts = set(force_timeout_txids)
    inboxes: dict[int, list[LFHEPACProposal]] = {node: [] for node in range(state.num_nodes)}
    for proposal in proposals:
        for endpoint in proposal.affected_endpoints:
            inboxes[endpoint].append(proposal)

    responses: list[EndpointResponse] = []
    local_winners: dict[int, str] = {}
    vetoes = {"initiator": 0, "removed_neighbor": 0, "candidate": 0}
    for endpoint in range(state.num_nodes):
        touching = inboxes[endpoint]
        if not touching:
            continue
        valid: list[LFHEPACProposal] = []
        invalid_reasons: dict[str, str] = {}
        for proposal in touching:
            reason = _topology_validation_reason(state, snapshot, proposal, endpoint)
            if reason is None:
                reason = _score_validation_reason(snapshot, proposal, endpoint, method)
            if reason is None:
                valid.append(proposal)
            else:
                invalid_reasons[proposal.txid] = reason
                if "score_veto" in reason:
                    if endpoint == proposal.initiator:
                        vetoes["initiator"] += 1
                    elif endpoint == proposal.removed_neighbor:
                        vetoes["removed_neighbor"] += 1
                    else:
                        vetoes["candidate"] += 1
        winner = (
            min(valid, key=lambda p: _endpoint_priority(p, method=method, seed=seed))
            if valid
            else None
        )
        if winner is not None:
            local_winners[endpoint] = winner.txid
            state.locks[endpoint] = winner.txid
        for proposal in touching:
            if proposal.txid in invalid_reasons:
                granted, reason = False, invalid_reasons[proposal.txid]
            elif winner is not None and proposal.txid == winner.txid:
                granted, reason = True, "grant"
            else:
                granted, reason = False, "local_arbitration_loss"
            responses.append(
                EndpointResponse(
                    txid=proposal.txid,
                    endpoint_id=endpoint,
                    granted=granted,
                    reason=reason,
                    topology_version=state.versions[endpoint],
                    evidence_hash=proposal.evidence_hash,
                )
            )

    by_txid = {proposal.txid: proposal for proposal in proposals}
    response_map: dict[str, list[EndpointResponse]] = {txid: [] for txid in by_txid}
    for response in responses:
        response_map[response.txid].append(response)
    certified: list[LFHEPACProposal] = []
    outcomes: list[TransactionOutcome] = []
    stale_rejections = 0
    conflict_rejections = 0
    timeout_aborts = 0
    for proposal in sorted(proposals, key=lambda item: item.txid):
        tx_responses = response_map[proposal.txid]
        if proposal.txid in timeouts:
            timeout_aborts += 1
            outcomes.append(TransactionOutcome(proposal.txid, False, "timeout"))
            continue
        if len(tx_responses) != len(proposal.affected_endpoints):
            timeout_aborts += 1
            outcomes.append(TransactionOutcome(proposal.txid, False, "missing_grant"))
            continue
        if not all(response.granted for response in tx_responses):
            reasons = {response.reason for response in tx_responses if not response.granted}
            if any("stale" in reason for reason in reasons):
                stale_rejections += 1
            if "local_arbitration_loss" in reasons:
                conflict_rejections += 1
            outcomes.append(
                TransactionOutcome(proposal.txid, False, "+".join(sorted(reasons)))
            )
            continue
        certified.append(proposal)

    committed_additions = 0
    committed_swaps = 0
    budget_rejections = 0
    already_recorded = {outcome.txid for outcome in outcomes}
    for index, proposal in enumerate(sorted(certified, key=lambda item: item.txid)):
        if index >= max_commits:
            budget_rejections += 1
            outcomes.append(TransactionOutcome(proposal.txid, False, "commit_budget_exhausted"))
            continue
        if _atomic_apply(state, proposal):
            if proposal.operation == "addition":
                committed_additions += 1
            else:
                committed_swaps += 1
            outcomes.append(TransactionOutcome(proposal.txid, True, "committed"))
        else:
            outcomes.append(TransactionOutcome(proposal.txid, False, "atomic_validation_failed"))
    for proposal in proposals:
        if proposal.txid not in already_recorded and proposal.txid not in {
            outcome.txid for outcome in outcomes
        }:
            outcomes.append(TransactionOutcome(proposal.txid, False, "abort"))
    state.locks.clear()
    state.validate()

    # Endpoint responses carry txid, endpoint, version, decision, reason code and hash.
    response_bytes = len(responses) * (8 + 8 + 8 + 1 + 8 + 32)
    proposal_bytes = sum(
        8 * (8 + len(proposal.affected_endpoints))
        + 8 * len(proposal.endpoint_gains)
        + 32
        for proposal in proposals
    )
    return PACEpochResult(
        outcomes=tuple(sorted(outcomes, key=lambda outcome: outcome.txid)),
        responses=tuple(sorted(responses, key=lambda r: (r.txid, r.endpoint_id))),
        committed_additions=committed_additions,
        committed_swaps=committed_swaps,
        conflict_rejections=conflict_rejections,
        stale_rejections=stale_rejections,
        timeout_aborts=timeout_aborts,
        budget_rejections=budget_rejections,
        endpoint_vetoes=tuple(sorted(vetoes.items())),
        control_messages=len(proposals) + len(responses),
        control_bytes=proposal_bytes + response_bytes,
    )


def proposal_log_rows(
    proposals: Sequence[LFHEPACProposal],
    result: PACEpochResult,
) -> list[dict[str, object]]:
    response_map: dict[str, list[EndpointResponse]] = {}
    for response in result.responses:
        response_map.setdefault(response.txid, []).append(response)
    outcomes = {outcome.txid: outcome for outcome in result.outcomes}
    rows: list[dict[str, object]] = []
    for proposal in proposals:
        outcome = outcomes[proposal.txid]
        rows.append(
            {
                "txid": proposal.txid,
                "initiator": proposal.initiator,
                "fof_path": proposal.fof_path,
                "removed_neighbor": proposal.removed_neighbor,
                "operation": proposal.operation,
                "affected_endpoints": proposal.affected_endpoints,
                "scores_before": dict(proposal.scores_before),
                "scores_after": dict(proposal.scores_after),
                "endpoint_gains": dict(proposal.endpoint_gains),
                "endpoint_responses": [response.__dict__ for response in response_map.get(proposal.txid, [])],
                "committed": outcome.committed,
                "final_reason": outcome.reason,
                "evidence_hash": proposal.evidence_hash,
            }
        )
    return rows
