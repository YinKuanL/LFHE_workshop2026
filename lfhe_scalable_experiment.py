#!/usr/bin/env python3
"""Scalable LFHE experiment runner for 1000-client simulation.

Key design choices
------------------
* Client model states live in CPU RAM.
* A single reusable working model performs local training/evaluation on GPU.
* Only active clients are trained and aggregated each round.
* Sparse bounded-degree graphs are used; no fully connected 1000-client run.
* Atomic checkpoints include model states, topology, metrics, and RNG states.
* SIGUSR1/SIGTERM request an immediate checkpoint and clean exit for SLURM requeue.

The implementation supports ring, static_random, and LFHE. It is intentionally
self-contained so that the 1000-client scalability run does not depend on the
older per-client-GPU implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

TensorState = Dict[str, torch.Tensor]

_STOP_REQUESTED = False
_STOP_SIGNAL: Optional[int] = None


def _request_stop(signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED, _STOP_SIGNAL
    _STOP_REQUESTED = True
    _STOP_SIGNAL = signum
    print(f"\n[signal] Received signal {signum}; checkpointing after the current safe step.", flush=True)


signal.signal(signal.SIGTERM, _request_stop)
if hasattr(signal, "SIGUSR1"):
    signal.signal(signal.SIGUSR1, _request_stop)


@dataclass
class Config:
    method: str
    num_clients: int
    rounds: int
    seed: int
    alpha: float
    min_samples: int
    participation_rate: float
    local_steps: int
    batch_size: int
    lr: float
    max_degree: int
    topology_interval: int
    eval_interval: int
    eval_clients: int
    checkpoint_interval: int
    output_dir: str
    data_root: str
    num_workers: int
    w1: float
    w2: float
    w3: float
    beta0: float
    beta_decay: float
    epsilon: float
    final_eval_all: bool


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.avgpool(self.features(x)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_dataset(data_root: str) -> Tuple[Any, Any]:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ])
    train = datasets.CIFAR10(data_root, train=True, download=True, transform=train_transform)
    test = datasets.CIFAR10(data_root, train=False, download=True, transform=test_transform)
    return train, test


def dirichlet_split_with_repair(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    min_samples: int,
    rng: np.random.Generator,
) -> List[List[int]]:
    """Class-wise Dirichlet split followed by deterministic minimum-size repair.

    The repair moves randomly selected examples from the largest clients to clients
    below ``min_samples``. This avoids empty DataLoaders at large N while preserving
    the original Dirichlet allocation as much as possible.
    """
    if min_samples * num_clients > len(labels):
        raise ValueError(
            f"min_samples*num_clients={min_samples*num_clients} exceeds dataset size {len(labels)}"
        )

    splits: List[List[int]] = [[] for _ in range(num_clients)]
    num_classes = int(labels.max()) + 1
    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
        counts = rng.multinomial(len(class_indices), proportions)
        cursor = 0
        for client_id, count in enumerate(counts.tolist()):
            if count:
                splits[client_id].extend(class_indices[cursor:cursor + count].tolist())
                cursor += count

    deficient = [i for i, idxs in enumerate(splits) if len(idxs) < min_samples]
    while deficient:
        receiver = deficient.pop()
        needed = min_samples - len(splits[receiver])
        for _ in range(needed):
            donors = [i for i, idxs in enumerate(splits) if len(idxs) > min_samples]
            if not donors:
                raise RuntimeError("Unable to repair split while respecting min_samples")
            donor = max(donors, key=lambda i: len(splits[i]))
            move_pos = int(rng.integers(0, len(splits[donor])))
            splits[receiver].append(splits[donor].pop(move_pos))

    for idxs in splits:
        rng.shuffle(idxs)
    return splits


def label_entropy(labels: np.ndarray, splits: Sequence[Sequence[int]]) -> float:
    entropies: List[float] = []
    for indices in splits:
        client_labels = labels[np.asarray(indices, dtype=np.int64)]
        counts = np.bincount(client_labels, minlength=int(labels.max()) + 1)
        probs = counts[counts > 0] / counts.sum()
        entropies.append(float(-(probs * np.log(probs)).sum()))
    return float(np.mean(entropies))


def clone_state_to_cpu(state: Mapping[str, torch.Tensor]) -> TensorState:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def initialize_client_states(num_clients: int, seed: int) -> List[TensorState]:
    """Initialize every client from exactly the same model state."""
    torch.manual_seed(seed)
    base_model = CNN().cpu()
    base_state = clone_state_to_cpu(base_model.state_dict())
    return [{k: v.clone() for k, v in base_state.items()} for _ in range(num_clients)]


def build_bounded_connected_graph(num_clients: int, max_degree: int, seed: int) -> nx.Graph:
    """Connected graph with max degree respected from round zero.

    Start from a randomly permuted ring and add bounded random chords until no
    additional feasible pair is found or the target average degree is reached.
    """
    if num_clients < 3:
        raise ValueError("num_clients must be at least 3")
    if max_degree < 2:
        raise ValueError("max_degree must be at least 2 for a connected ring")

    rng = random.Random(seed)
    order = list(range(num_clients))
    rng.shuffle(order)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_clients))
    for pos, node in enumerate(order):
        graph.add_edge(node, order[(pos + 1) % num_clients])

    target_edges = (num_clients * max_degree) // 2
    attempts = 0
    max_attempts = max(10_000, num_clients * max_degree * 50)
    while graph.number_of_edges() < target_edges and attempts < max_attempts:
        u, v = rng.sample(range(num_clients), 2)
        attempts += 1
        if graph.has_edge(u, v):
            continue
        if graph.degree(u) >= max_degree or graph.degree(v) >= max_degree:
            continue
        graph.add_edge(u, v)
    return graph


def build_graph(method: str, num_clients: int, max_degree: int, seed: int) -> nx.Graph:
    if method == "ring":
        return nx.cycle_graph(num_clients)
    if method in {"static_random", "lfhe"}:
        return build_bounded_connected_graph(num_clients, max_degree, seed)
    raise ValueError(f"Unsupported method: {method}")


def sample_local_batch(
    dataset: Any,
    client_indices: Sequence[int],
    batch_size: int,
    rng: random.Random,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not client_indices:
        raise ValueError("Client has no samples")
    if len(client_indices) >= batch_size:
        chosen = rng.sample(list(client_indices), batch_size)
    else:
        chosen = [rng.choice(client_indices) for _ in range(batch_size)]
    examples = [dataset[index] for index in chosen]
    x = torch.stack([item[0] for item in examples], dim=0)
    y = torch.tensor([int(item[1]) for item in examples], dtype=torch.long)
    return x, y


def train_one_client(
    working_model: nn.Module,
    initial_state: TensorState,
    dataset: Any,
    indices: Sequence[int],
    device: torch.device,
    lr: float,
    batch_size: int,
    local_steps: int,
    rng: random.Random,
) -> TensorState:
    working_model.load_state_dict(initial_state, strict=True)
    working_model.to(device)
    working_model.train()
    optimizer = optim.SGD(working_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_steps):
        x, y = sample_local_batch(dataset, indices, batch_size, rng)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(working_model(x), y)
        loss.backward()
        optimizer.step()
    state = clone_state_to_cpu(working_model.state_dict())
    working_model.cpu()
    return state


def weighted_state_average(weighted_states: Sequence[Tuple[float, TensorState]]) -> TensorState:
    """Average floating tensors; preserve integer buffers from the self model."""
    if not weighted_states:
        raise ValueError("weighted_states cannot be empty")
    output: TensorState = {}
    reference = weighted_states[-1][1]
    for key, ref_tensor in reference.items():
        if ref_tensor.is_floating_point() or ref_tensor.is_complex():
            accumulator = torch.zeros_like(ref_tensor, dtype=torch.float32)
            for weight, state in weighted_states:
                accumulator.add_(state[key].float(), alpha=float(weight))
            output[key] = accumulator.to(dtype=ref_tensor.dtype)
        else:
            output[key] = ref_tensor.clone()
    return output


def aggregate_active_clients(
    client_states: List[TensorState],
    graph: nx.Graph,
    active_clients: Sequence[int],
) -> None:
    active_set = set(active_clients)
    updates: Dict[int, TensorState] = {}
    for client_id in active_clients:
        active_neighbors = [n for n in graph.neighbors(client_id) if n in active_set]
        weighted: List[Tuple[float, TensorState]] = []
        neighbor_weights: List[float] = []
        for neighbor in active_neighbors:
            weight = 1.0 / (1.0 + max(graph.degree(client_id), graph.degree(neighbor)))
            neighbor_weights.append(weight)
            weighted.append((weight, client_states[neighbor]))
        self_weight = 1.0 - sum(neighbor_weights)
        if self_weight < -1e-7:
            raise RuntimeError(f"Negative self weight for client {client_id}: {self_weight}")
        weighted.append((max(self_weight, 0.0), client_states[client_id]))
        updates[client_id] = weighted_state_average(weighted)
    for client_id, state in updates.items():
        client_states[client_id] = state


def extract_representations(client_states: Sequence[TensorState]) -> torch.Tensor:
    """Use flattened final classifier weights (10 x 256 = 2560 dimensions)."""
    key = "classifier.4.weight"
    return torch.stack([state[key].float().reshape(-1) for state in client_states], dim=0)


def cosine_distance_to_neighbor_mean(
    client_id: int,
    graph: nx.Graph,
    representations: torch.Tensor,
) -> float:
    neighbors = list(graph.neighbors(client_id))
    if not neighbors:
        return 0.0
    center = representations[client_id]
    mean_neighbor = representations[neighbors].mean(dim=0)
    denom = torch.linalg.vector_norm(center) * torch.linalg.vector_norm(mean_neighbor)
    if float(denom) <= 1e-12:
        return 0.0
    cosine = torch.dot(center, mean_neighbor) / denom
    return float(1.0 - cosine.clamp(-1.0, 1.0))


def local_fitness(
    client_id: int,
    graph: nx.Graph,
    representations: torch.Tensor,
    round_index: int,
    max_degree: int,
    w1: float,
    w2: float,
    w3: float,
    beta0: float,
    beta_decay: float,
) -> float:
    neighbors = list(graph.neighbors(client_id))
    if neighbors:
        diff = representations[neighbors] - representations[client_id]
        structural = float(torch.linalg.vector_norm(diff, dim=1).sum())
    else:
        structural = 0.0
    mismatch = cosine_distance_to_neighbor_mean(client_id, graph, representations)
    beta = beta0 * math.exp(-beta_decay * round_index)
    degree_penalty = graph.degree(client_id) / max_degree
    return w1 * (1.0 - beta) * structural + w2 * beta * mismatch - w3 * degree_penalty


def has_local_alternative_path(graph: nx.Graph, u: int, v: int) -> bool:
    return bool(set(graph.neighbors(u)).intersection(graph.neighbors(v)))


def lfhe_update_scalable(
    graph: nx.Graph,
    client_states: Sequence[TensorState],
    round_index: int,
    active_clients: Sequence[int],
    max_degree: int,
    w1: float,
    w2: float,
    w3: float,
    beta0: float,
    beta_decay: float,
    epsilon: float,
    rng: random.Random,
) -> Tuple[nx.Graph, Dict[str, int]]:
    """Sequential local LFHE update matching the submitted algorithmic behavior."""
    updated = graph.copy()
    representations = extract_representations(client_states)
    accepted = 0
    candidate_checks = 0

    order = list(active_clients)
    rng.shuffle(order)
    for client_id in order:
        neighbors = list(updated.neighbors(client_id))
        if not neighbors:
            continue
        intermediary = rng.choice(neighbors)
        candidates = [
            node for node in updated.neighbors(intermediary)
            if node != client_id and not updated.has_edge(client_id, node)
        ]
        if not candidates:
            continue
        candidate = rng.choice(candidates)
        candidate_checks += 1
        old_fitness = local_fitness(
            client_id, updated, representations, round_index, max_degree,
            w1, w2, w3, beta0, beta_decay,
        )

        if updated.degree(client_id) < max_degree and updated.degree(candidate) < max_degree:
            updated.add_edge(client_id, candidate)
            new_fitness = local_fitness(
                client_id, updated, representations, round_index, max_degree,
                w1, w2, w3, beta0, beta_decay,
            )
            if new_fitness > old_fitness + epsilon:
                accepted += 1
            else:
                updated.remove_edge(client_id, candidate)
            continue

        if updated.degree(candidate) >= max_degree or updated.degree(client_id) <= 1:
            continue

        best_neighbor: Optional[int] = None
        best_fitness = -float("inf")
        for removed_neighbor in neighbors:
            if updated.degree(removed_neighbor) <= 1:
                continue
            if not has_local_alternative_path(updated, client_id, removed_neighbor):
                continue
            updated.remove_edge(client_id, removed_neighbor)
            updated.add_edge(client_id, candidate)
            swap_fitness = local_fitness(
                client_id, updated, representations, round_index, max_degree,
                w1, w2, w3, beta0, beta_decay,
            )
            updated.remove_edge(client_id, candidate)
            updated.add_edge(client_id, removed_neighbor)
            if swap_fitness > best_fitness:
                best_fitness = swap_fitness
                best_neighbor = removed_neighbor

        if best_neighbor is not None and best_fitness > old_fitness + epsilon:
            updated.remove_edge(client_id, best_neighbor)
            updated.add_edge(client_id, candidate)
            accepted += 1

    assert_graph_invariants(updated, max_degree)
    return updated, {"candidate_checks": candidate_checks, "accepted_rewires": accepted}


def assert_graph_invariants(graph: nx.Graph, max_degree: int) -> None:
    if not nx.is_connected(graph):
        raise RuntimeError("Topology became disconnected")
    over_cap = [node for node, degree in graph.degree() if degree > max_degree]
    if over_cap:
        raise RuntimeError(f"Degree cap violated by nodes: {over_cap[:10]}")
    if any(u == v for u, v in graph.edges()):
        raise RuntimeError("Self-loop detected")


def evaluate_clients(
    working_model: nn.Module,
    client_states: Sequence[TensorState],
    client_ids: Sequence[int],
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, List[float]]:
    criterion = nn.CrossEntropyLoss(reduction="sum")
    accuracies: List[float] = []
    losses: List[float] = []
    for client_id in client_ids:
        working_model.load_state_dict(client_states[client_id], strict=True)
        working_model.to(device)
        working_model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        with torch.inference_mode():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = working_model(x)
                total_loss += float(criterion(logits, y))
                correct += int((logits.argmax(dim=1) == y).sum())
                total += int(y.numel())
        accuracies.append(correct / total)
        losses.append(total_loss / total)
        working_model.cpu()
    return float(np.mean(accuracies)), float(np.mean(losses)), accuracies


def graph_to_edge_array(graph: nx.Graph) -> np.ndarray:
    return np.asarray(list(graph.edges()), dtype=np.int32)


def graph_from_edge_array(num_clients: int, edges: np.ndarray) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(num_clients))
    graph.add_edges_from((int(u), int(v)) for u, v in edges.tolist())
    return graph


def atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, destination)


def capture_rng_state(local_rng: random.Random) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python_global": random.getstate(),
        "python_local": local_rng.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any], local_rng: random.Random) -> None:
    random.setstate(state["python_global"])
    local_rng.setstate(state["python_local"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: Path,
    config: Config,
    next_round: int,
    client_states: Sequence[TensorState],
    graph: nx.Graph,
    splits: Sequence[Sequence[int]],
    metrics: Mapping[str, Any],
    local_rng: random.Random,
    reason: str,
) -> None:
    started = time.perf_counter()
    payload = {
        "format_version": 1,
        "config": asdict(config),
        "next_round": next_round,
        "client_states": list(client_states),
        "graph_edges": graph_to_edge_array(graph),
        "splits": [list(indices) for indices in splits],
        "metrics": dict(metrics),
        "rng_state": capture_rng_state(local_rng),
        "reason": reason,
        "saved_at_unix": time.time(),
    }
    atomic_torch_save(payload, path)
    print(
        f"[checkpoint] Saved {path} at next_round={next_round} "
        f"({time.perf_counter() - started:.1f}s, reason={reason})",
        flush=True,
    )


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    print(f"[checkpoint] Loading {path}", flush=True)
    return torch.load(path, map_location="cpu", weights_only=False)


def write_metrics(metrics: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "metrics.json.tmp"
    final = output_dir / "metrics.json"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    os.replace(temporary, final)


def select_active_clients(num_clients: int, participation_rate: float, rng: random.Random) -> List[int]:
    count = max(1, min(num_clients, int(round(num_clients * participation_rate))))
    return sorted(rng.sample(range(num_clients), count))


def run(config: Config, resume: bool) -> int:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}", flush=True)
    if device.type != "cuda":
        print("[warning] CUDA is unavailable; the experiment will run on CPU.", flush=True)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    completed_path = output_dir / "SUCCESS"
    if completed_path.exists():
        print(f"[skip] SUCCESS already exists at {completed_path}")
        return 0

    train_set, test_set = load_dataset(config.data_root)
    labels = np.asarray(train_set.targets, dtype=np.int64)
    test_loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )

    local_rng = random.Random(config.seed + 17)
    if resume and checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, device)
        stored_config = checkpoint["config"]
        immutable_keys = ["method", "num_clients", "seed", "alpha", "max_degree"]
        for key in immutable_keys:
            if stored_config[key] != getattr(config, key):
                raise ValueError(
                    f"Checkpoint mismatch for {key}: {stored_config[key]} != {getattr(config, key)}"
                )
        start_round = int(checkpoint["next_round"])
        client_states = checkpoint["client_states"]
        graph = graph_from_edge_array(config.num_clients, checkpoint["graph_edges"])
        splits = checkpoint["splits"]
        metrics = checkpoint["metrics"]
        restore_rng_state(checkpoint["rng_state"], local_rng)
        print(f"[resume] Continuing from round {start_round}", flush=True)
    else:
        split_rng = np.random.default_rng(config.seed)
        splits = dirichlet_split_with_repair(
            labels, config.num_clients, config.alpha, config.min_samples, split_rng
        )
        print(
            f"[data] clients={config.num_clients}, min={min(map(len, splits))}, "
            f"max={max(map(len, splits))}, mean={np.mean(list(map(len, splits))):.2f}, "
            f"entropy={label_entropy(labels, splits):.4f}",
            flush=True,
        )
        client_states = initialize_client_states(config.num_clients, config.seed)
        graph = build_graph(config.method, config.num_clients, config.max_degree, config.seed)
        assert_graph_invariants(graph, config.max_degree)
        start_round = 0
        metrics: Dict[str, Any] = {
            "round": [],
            "mean_accuracy": [],
            "mean_loss": [],
            "evaluated_clients": [],
            "round_seconds": [],
            "train_seconds": [],
            "aggregation_seconds": [],
            "topology_seconds": [],
            "candidate_checks": [],
            "accepted_rewires": [],
            "active_clients": [],
            "active_edges": [],
        }

    working_model = CNN().cpu()
    evaluation_rng = random.Random(config.seed + 991)
    fixed_eval_count = min(config.eval_clients, config.num_clients)
    fixed_eval_clients = sorted(evaluation_rng.sample(range(config.num_clients), fixed_eval_count))

    for round_index in range(start_round, config.rounds):
        round_started = time.perf_counter()
        active_clients = select_active_clients(
            config.num_clients, config.participation_rate, local_rng
        )

        train_started = time.perf_counter()
        for client_id in active_clients:
            client_states[client_id] = train_one_client(
                working_model=working_model,
                initial_state=client_states[client_id],
                dataset=train_set,
                indices=splits[client_id],
                device=device,
                lr=config.lr,
                batch_size=config.batch_size,
                local_steps=config.local_steps,
                rng=local_rng,
            )
            if _STOP_REQUESTED:
                # Finish the current client update, then checkpoint a consistent state.
                break
        train_seconds = time.perf_counter() - train_started

        if _STOP_REQUESTED:
            save_checkpoint(
                checkpoint_path, config, round_index, client_states, graph, splits,
                metrics, local_rng, reason=f"signal_{_STOP_SIGNAL}_during_training",
            )
            return 99

        aggregation_started = time.perf_counter()
        aggregate_active_clients(client_states, graph, active_clients)
        aggregation_seconds = time.perf_counter() - aggregation_started

        topology_seconds = 0.0
        topo_stats = {"candidate_checks": 0, "accepted_rewires": 0}
        if config.method == "lfhe" and round_index % config.topology_interval == 0:
            topology_started = time.perf_counter()
            graph, topo_stats = lfhe_update_scalable(
                graph=graph,
                client_states=client_states,
                round_index=round_index,
                active_clients=active_clients,
                max_degree=config.max_degree,
                w1=config.w1,
                w2=config.w2,
                w3=config.w3,
                beta0=config.beta0,
                beta_decay=config.beta_decay,
                epsilon=config.epsilon,
                rng=local_rng,
            )
            topology_seconds = time.perf_counter() - topology_started

        round_seconds = time.perf_counter() - round_started
        metrics["round_seconds"].append(round_seconds)
        metrics["train_seconds"].append(train_seconds)
        metrics["aggregation_seconds"].append(aggregation_seconds)
        metrics["topology_seconds"].append(topology_seconds)
        metrics["candidate_checks"].append(topo_stats["candidate_checks"])
        metrics["accepted_rewires"].append(topo_stats["accepted_rewires"])
        metrics["active_clients"].append(len(active_clients))
        metrics["active_edges"].append(graph.number_of_edges())

        should_evaluate = (
            round_index % config.eval_interval == 0 or round_index == config.rounds - 1
        )
        if should_evaluate:
            eval_ids = (
                list(range(config.num_clients))
                if config.final_eval_all and round_index == config.rounds - 1
                else fixed_eval_clients
            )
            mean_acc, mean_loss, _ = evaluate_clients(
                working_model, client_states, eval_ids, test_loader, device
            )
            metrics["round"].append(round_index)
            metrics["mean_accuracy"].append(mean_acc)
            metrics["mean_loss"].append(mean_loss)
            metrics["evaluated_clients"].append(len(eval_ids))
            print(
                f"[{config.method} seed={config.seed}] round={round_index:04d} "
                f"acc={mean_acc:.4f} loss={mean_loss:.4f} active={len(active_clients)} "
                f"edges={graph.number_of_edges()} rewires={topo_stats['accepted_rewires']} "
                f"time={round_seconds:.1f}s",
                flush=True,
            )
            write_metrics(metrics, output_dir)

        if (round_index + 1) % config.checkpoint_interval == 0:
            save_checkpoint(
                checkpoint_path, config, round_index + 1, client_states, graph,
                splits, metrics, local_rng, reason="periodic",
            )

        if _STOP_REQUESTED:
            save_checkpoint(
                checkpoint_path, config, round_index + 1, client_states, graph,
                splits, metrics, local_rng, reason=f"signal_{_STOP_SIGNAL}",
            )
            return 99

    save_checkpoint(
        checkpoint_path, config, config.rounds, client_states, graph, splits,
        metrics, local_rng, reason="completed",
    )
    write_metrics(metrics, output_dir)
    completed_path.write_text("SUCCESS\n", encoding="utf-8")
    print(f"[done] Wrote {completed_path}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["ring", "static_random", "lfhe"], default="lfhe")
    parser.add_argument("--num-clients", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--participation-rate", type=float, default=0.1)
    parser.add_argument("--local-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--max-degree", type=int, default=4)
    parser.add_argument("--topology-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-clients", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default=os.environ.get("LFHE_DATA_ROOT", "./data"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=1.0)
    parser.add_argument("--w3", type=float, default=0.1)
    parser.add_argument("--beta0", type=float, default=1.0)
    parser.add_argument("--beta-decay", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--final-eval-all", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.participation_rate <= 1.0:
        raise ValueError("participation_rate must be in (0, 1]")
    if args.local_steps < 1:
        raise ValueError("local_steps must be >= 1")
    config = Config(
        method=args.method,
        num_clients=args.num_clients,
        rounds=args.rounds,
        seed=args.seed,
        alpha=args.alpha,
        min_samples=args.min_samples,
        participation_rate=args.participation_rate,
        local_steps=args.local_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        max_degree=args.max_degree,
        topology_interval=args.topology_interval,
        eval_interval=args.eval_interval,
        eval_clients=args.eval_clients,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=args.output_dir,
        data_root=args.data_root,
        num_workers=args.num_workers,
        w1=args.w1,
        w2=args.w2,
        w3=args.w3,
        beta0=args.beta0,
        beta_decay=args.beta_decay,
        epsilon=args.epsilon,
        final_eval_all=args.final_eval_all,
    )
    return run(config, resume=args.resume)


if __name__ == "__main__":
    sys.exit(main())
