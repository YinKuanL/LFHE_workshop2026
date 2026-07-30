"""Canonical LFHE topology policy used by the reported CIFAR-10 experiments.

Keep scientific changes out of this module.  Execution instrumentation is exposed
through the optional ``trace`` argument and does not alter random draws or decisions.
"""
import random
import numpy as np
import torch


def anneal_beta(round, beta0=1.0, kappa=0.01):
    return beta0 * np.exp(-kappa * round)


def local_laplacian_proxy(graph, i, clients):
    wi = clients[i].model.get_representation()
    return sum((torch.sum((wi - clients[j].model.get_representation()) ** 2)
                for j in graph.neighbors(i)), torch.tensor(0.0, device=wi.device))


def neighborhood_similarity(graph, i, clients):
    neighbors = list(graph.neighbors(i))
    if not neighbors:
        return 0.0
    wi = clients[i].model.get_representation()
    mean = torch.stack([clients[j].model.get_representation() for j in neighbors]).mean(0)
    return torch.nn.functional.cosine_similarity(wi, mean, dim=0).item()


def compute_fitness(i, graph, clients, D_max, w1=2.0, w2=1.0, w3=0.1, round=0):
    beta = anneal_beta(round)
    return (w1 * (1 - beta) * local_laplacian_proxy(graph, i, clients)
            + w2 * beta * (1 - neighborhood_similarity(graph, i, clients))
            - w3 * graph.degree[i] / max(D_max, 1))


def has_alternative_path(graph, i, j):
    return bool(set(graph.neighbors(i)).intersection(graph.neighbors(j)))


def lfhe_update(graph, clients, epsilon=0.05, D_max=5, w1=1.0, w2=1.0,
                w3=0.1, round=0, trace=None, client_ids=None):
    """Canonical sequential update; epsilon is retained for API compatibility.

    ``client_ids`` only scopes scalable/partial deployment.  Its default preserves
    the canonical 0..N-1 update order exactly.
    """
    graph = graph.copy()
    ids = range(len(clients)) if client_ids is None else client_ids
    for i in ids:
        event = {"client": int(i), "candidate_checks": 0,
                 "fitness_evaluations": 0, "action": "no_candidate"}
        neighbors = list(graph.neighbors(i))
        if not neighbors:
            if trace is not None: trace.append(event)
            continue
        intermediary = random.choice(neighbors)
        candidates = [k for k in graph.neighbors(intermediary)
                      if k != i and not graph.has_edge(i, k)]
        if not candidates:
            if trace is not None: trace.append(event)
            continue
        candidate = random.choice(candidates)
        event.update(candidate_checks=1, fitness_evaluations=1,
                     candidate=int(candidate), action="rejected_proposal")
        old = compute_fitness(i, graph, clients, D_max, w1, w2, w3, round)
        if graph.degree(i) < D_max and graph.degree(candidate) < D_max:
            graph.add_edge(i, candidate)
            new = compute_fitness(i, graph, clients, D_max, w1, w2, w3, round)
            event["fitness_evaluations"] += 1
            if new > old:
                event["action"] = "accepted_addition"
            else:
                graph.remove_edge(i, candidate)
        else:
            best, remove = -float("inf"), None
            for neighbor in list(graph.neighbors(i)):
                if graph.degree(i) <= 1 or graph.degree(candidate) >= D_max:
                    continue
                if graph.degree(neighbor) <= 1 or not has_alternative_path(graph, i, neighbor):
                    continue
                graph.remove_edge(i, neighbor); graph.add_edge(i, candidate)
                score = compute_fitness(i, graph, clients, D_max, w1, w2, w3, round)
                event["candidate_checks"] += 1; event["fitness_evaluations"] += 1
                graph.remove_edge(i, candidate); graph.add_edge(i, neighbor)
                if score > best: best, remove = score, neighbor
            if remove is not None and best > old:
                graph.remove_edge(i, remove); graph.add_edge(i, candidate)
                event["action"] = "accepted_swap"
        if trace is not None: trace.append(event)
    return graph
