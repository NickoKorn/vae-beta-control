"""Ungewichtete Union-Struktur; Reciprocity auf dem gerichteten Graph.

Transitivity = 3 * Dreiecke / verbundene Tripel; mittleres Clustering gewichtet
Knoten anders. Keine allgemeine Güterichtung; k und Gradverteilung beeinflussen
beide. Definition: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.transitivity.html
"""
import networkx as nx
import numpy as np


def compute_neighborhood(graph):
    g = graph.undirected
    degrees = np.array([d for _, d in g.degree()], dtype=float)
    return dict(reciprocity=nx.reciprocity(graph.directed),
                clustering=nx.average_clustering(g), transitivity=nx.transitivity(g),
                degree_mean=degrees.mean(), degree_std=degrees.std(),
                degree_min=degrees.min(), degree_max=degrees.max())
