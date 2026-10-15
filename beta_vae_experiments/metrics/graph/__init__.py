"""Offline preservation of independently constructed kNN graph-geodesic rankings."""
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr


def _graph_distances(points, n_neighbors):
    graph = kneighbors_graph(points, n_neighbors=n_neighbors, mode='distance',
                             metric='euclidean', include_self=False)
    # Union: either directed neighbor selection creates an undirected edge.
    # Maximum preserves the selected Euclidean weight (not a half-weight).
    # Keep connectivity separately so coincident samples retain zero-cost edges.
    edges = graph.copy()
    edges.data = np.ones_like(edges.data)
    rows, cols = edges.maximum(edges.T).nonzero()
    weights = np.asarray(graph.maximum(graph.T)[rows, cols]).ravel()
    symmetric = csr_matrix((weights, (rows, cols)), shape=graph.shape)
    return shortest_path(symmetric, directed=False, method='D')


def compute_geodesic_preservation(X: np.ndarray, mu: np.ndarray,
                                  n_neighbors: int = 10) -> dict:
    """Compare shortest-path distance rankings on independent X and mu kNN graphs.

    Measures graph-geodesic ranking preservation, not direct Euclidean distances
    or exact neighbor identities. Inputs must have identical sample order; this
    cannot be verified from arrays alone. No scaling or resampling is performed.

    Graphs use Euclidean edge weights and union symmetrization: either endpoint
    may select an edge. Explicit zero-distance edges are retained. Only i < j
    pairs finite in BOTH graphs enter SciPy Spearman; disconnected pairs receive
    no penalty. Connectivity and usable pair coverage are reported separately.
    Correlation is NaN for fewer than two valid pairs or constant pair distances.

    n_neighbors controls the local graph scale. Default 10; values such as 5 or
    15 are predefined sensitivity parameters, never tuned for a desired result.
    All-pairs distance matrices require O(n_samples**2) memory.
    """
    X, mu = np.asarray(X), np.asarray(mu)
    for name, points in [('X', X), ('mu', mu)]:
        if points.ndim != 2 or points.shape[1] == 0:
            raise ValueError(f'{name} must be 2-D with at least one feature')
        if not np.issubdtype(points.dtype, np.number) or np.iscomplexobj(points):
            raise ValueError(f'{name} must contain real numeric values')
        if not np.isfinite(points).all():
            raise ValueError(f'{name} must contain only finite values')
    if len(X) != len(mu):
        raise ValueError('X and mu must have the same number of samples in the same order')
    n = len(X)
    if isinstance(n_neighbors, (bool, np.bool_)) or not isinstance(n_neighbors, (int, np.integer)) or not 1 <= n_neighbors < n:
        raise ValueError('n_neighbors must be an integer with 1 <= n_neighbors < n_samples')
    dx = _graph_distances(X.astype(np.float64), int(n_neighbors))
    dm = _graph_distances(mu.astype(np.float64), int(n_neighbors))
    upper = np.triu_indices(n, k=1)
    a, b = dx[upper], dm[upper]
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    rho = float('nan')
    if len(a) >= 2 and np.any(a != a[0]) and np.any(b != b[0]):
        rho = float(spearmanr(a, b).statistic)
    return {'geodesic_spearman': rho, 'n_neighbors': int(n_neighbors),
            'n_samples': n, 'n_total_pairs': len(valid),
            'n_valid_pairs': int(valid.sum()), 'valid_pair_fraction': float(valid.mean()),
            'x_graph_connected': bool(np.isfinite(dx).all()),
            'mu_graph_connected': bool(np.isfinite(dm).all())}
