"""Canonical sklearn kNN and NetworkX union graphs; no repair or resampling.

Distances are original representation distances, not affinities. Ties follow
sklearn's deterministic ordering for the supplied row order.

Ein Pfeil i→j bedeutet: j ist unter den k Nachbarn von i. Im Union-Graph reicht
eine Richtung für eine Kante; er ist kein mutual-kNN-Graph. Strukturmetriken
sind ungewichtet. Gleiche IDs/k verhindern keine Tie-Artefakte. Kein Jitter
oder automatisches Reduzieren von k.
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
"""
from dataclasses import dataclass
import numpy as np
import networkx as nx
import pandas as pd
from sklearn.neighbors import NearestNeighbors


@dataclass
class RepresentationGraph:
    directed: nx.DiGraph
    undirected: nx.Graph
    neighbors: object
    sample_ids: np.ndarray
    k: int


def build_graph(values, sample_ids, k=20, metric="euclidean"):
    """Accept features or a symmetric distance matrix with metric='precomputed'."""
    if np.iscomplexobj(values):
        raise ValueError("Representation must be real-valued")
    values = np.asarray(values, dtype=float)
    ids = np.asarray(sample_ids)
    if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("Representation must be a finite nonempty feature matrix")
    n = len(values)
    if ids.ndim != 1 or len(ids) != n or pd.isna(ids).any() or len(set(map(str, ids))) != n:
        raise ValueError("sample_ids must be unique and aligned with all rows")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 1 <= k < n:
        raise ValueError("Require integer 1 <= k < n; k is never reduced")
    if metric == "precomputed":
        if values.shape != (n, n) or (values < 0).any() or not np.allclose(values, values.T) or not np.allclose(values.diagonal(), 0):
            raise ValueError("Precomputed distances must be square, symmetric, nonnegative with zero diagonal")
    nn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=1).fit(values)
    neighbors = nn.kneighbors_graph(X=None, mode="distance")
    directed = nx.DiGraph()
    directed.add_nodes_from((i, {"sample_id": str(s)}) for i, s in enumerate(ids))
    for i in range(n):
        for pos in range(neighbors.indptr[i], neighbors.indptr[i + 1]):
            directed.add_edge(i, int(neighbors.indices[pos]), distance=float(neighbors.data[pos]))
    union = directed.to_undirected()
    for i, j in union.edges:
        distances = [directed[a][b]["distance"] for a, b in ((i, j), (j, i)) if directed.has_edge(a, b)]
        union[i][j].update(distance=float(np.mean(distances)), mutual=len(distances) == 2)
    return RepresentationGraph(directed, union, neighbors, ids.copy(), int(k))
