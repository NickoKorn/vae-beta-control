"""Unweighted igraph Leiden modularity, fixed resolution 1, no labels.

RNG access is serialized; igraph's Python default RNG is restored afterward.
See graph_metrics.md and Traag et al., Scientific Reports (2019).
"""
import random
from threading import Lock
import numpy as np

_RNG_LOCK = Lock()


def compute_communities(graph, seed=31):
    import igraph as ig
    g = ig.Graph(n=len(graph.sample_ids), edges=list(graph.undirected.edges), directed=False)
    with _RNG_LOCK:
        ig.set_random_number_generator(random.Random(seed))
        try:
            partition = g.community_leiden(objective_function="modularity", resolution=1.0, n_iterations=2)
        finally:
            ig.set_random_number_generator(random)
    sizes = np.asarray(partition.sizes(), dtype=float)
    probabilities = sizes / sizes.sum()
    return dict(communities=len(sizes), modularity=partition.modularity,
                community_size_mean=sizes.mean(), community_size_median=np.median(sizes),
                community_size_min=sizes.min(), community_size_max=sizes.max(),
                largest_community_fraction=probabilities.max(),
                community_entropy=-np.sum(probabilities*np.log(probabilities))), {
                    "community": np.asarray(partition.membership, dtype=int)}
