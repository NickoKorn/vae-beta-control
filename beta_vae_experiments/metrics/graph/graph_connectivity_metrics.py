"""Whole-union connectivity via NetworkX; disconnected lambda2 is zero.

Unweighted algebraic_connectivity uses the library solver, seed 31 by default.
"""
import networkx as nx


def compute_connectivity(graph, seed=31):
    g = graph.undirected
    components = list(nx.connected_components(g))
    isolates = nx.number_of_isolates(g)
    return dict(components=len(components), giant_component_fraction=max(map(len, components))/len(g),
                isolated_nodes=isolates, isolated_fraction=isolates/len(g),
                lambda2=nx.algebraic_connectivity(g, weight=None, seed=seed))
