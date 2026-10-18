"""Unweighted igraph Leiden modularity, fixed resolution 1, no labels.

RNG access is serialized; igraph's Python default RNG is restored afterward.
Traag et al. (2019): https://www.nature.com/articles/s41598-019-41695-z
Leiden optimiert eine Qualitätsfunktion, keine biologische Wahrheit.
Auflösung 1 ist eine feste Referenz, kein bewiesenes Optimum. k/Auflösung/Seed
beeinflussen Partitionen; hohe Modularity allein belegt keine Preservation.
"""
import random
from threading import Lock
import numpy as np

_RNG_LOCK = Lock()


def compute_communities(graph, seed=31, resolution=1.0, n_iterations=2):
    import igraph as ig
    g = ig.Graph(n=len(graph.sample_ids), edges=list(graph.undirected.edges), directed=False)
    with _RNG_LOCK:
        ig.set_random_number_generator(random.Random(seed))
        try:
            partition = g.community_leiden(objective_function="modularity", resolution=resolution, n_iterations=n_iterations)
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
