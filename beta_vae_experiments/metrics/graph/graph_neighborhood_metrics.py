"""Label-free internal organization via NetworkX; see graph_metrics.md."""
import networkx as nx
import numpy as np


def compute_neighborhood(graph):
    g = graph.undirected
    degrees = np.array([d for _, d in g.degree()], dtype=float)
    return dict(reciprocity=nx.reciprocity(graph.directed),
                clustering=nx.average_clustering(g), transitivity=nx.transitivity(g),
                degree_mean=degrees.mean(), degree_std=degrees.std(),
                degree_min=degrees.min(), degree_max=degrees.max())
