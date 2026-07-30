"""Canonical epidemic baseline."""
import random
import networkx as nx

def build_epidemic_graph(num_clients, s=4, seed=None):
    rng = random.Random(seed); graph = nx.DiGraph(); graph.add_nodes_from(range(num_clients))
    for i in range(num_clients):
        peers = [j for j in range(num_clients) if j != i]
        graph.add_edges_from((i, j) for j in rng.sample(peers, min(s, len(peers))))
    return graph
