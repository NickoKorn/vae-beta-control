"""NetworkX unweighted paths and bottlenecks; see graph_metrics.md.

ASPL/diameter use the LCC (first component in node order breaks size ties).
Efficiency and exact normalized betweenness use the whole graph, including
disconnected nodes. A declared size guard fails this family, never subsamples.
"""
import networkx as nx
import numpy as np


def compute_paths(graph, max_nodes=3000):
    g = graph.undirected
    if len(g) > max_nodes:
        raise ValueError(f"Exact path metrics guard: {len(g)} > {max_nodes} nodes")
    lcc = g.subgraph(max(nx.connected_components(g), key=len))
    nodes = nx.betweenness_centrality(g, normalized=True, weight=None)
    edges = nx.edge_betweenness_centrality(g, normalized=True, weight=None)
    values = np.asarray([nodes[i] for i in g])
    metrics = dict(lcc_average_path_length=nx.average_shortest_path_length(lcc),
                   lcc_diameter=nx.diameter(lcc, usebounds=True),
                   global_efficiency=nx.global_efficiency(g),
                   betweenness_mean=values.mean(), betweenness_median=np.median(values),
                   betweenness_q90=np.quantile(values, .9), betweenness_max=values.max())
    return metrics, {"betweenness": values}, {"edge_betweenness": edges}
