"""Canonical epidemic baseline."""
import random
import networkx as nx

def build_epidemic_graph(num_clients, s=4, seed=None):
    rng = random.Random(seed); graph = nx.DiGraph(); graph.add_nodes_from(range(num_clients))
    for i in range(num_clients):
        peers = [j for j in range(num_clients) if j != i]
        graph.add_edges_from((i, j) for j in rng.sample(peers, min(s, len(peers))))
    return graph

def directed_aggregation(clients, graph):
    """
    Directed aggregation:
    node i receives from predecessors of i, plus itself.
    """
    new_states = []

    for i, c in enumerate(clients):
        incoming = list(graph.predecessors(i))
        models = [clients[j].model.state_dict() for j in incoming]
        models.append(c.model.state_dict())

        avg_state = {}
        for k in models[0]:
            avg_state[k] = sum(m[k] for m in models) / len(models)
        new_states.append(avg_state)

    for c, s in zip(clients, new_states):
        c.model.load_state_dict(s)
