"""Library Forman-Ricci (1d) curvature on the unit-weight kNN union graph.

Uses GraphRicciCurvature.FormanRicci, as in the historical graph analysis:
https://github.com/saibalmars/GraphRicciCurvature
For unit weights F(i,j) = 4 - degree(i) - degree(j). This describes graph
connectivity, not manifold curvature or a guarantee of downstream quality.
Representation distances are retained separately and are not edge weights.
"""
import numpy as np


def compute_curvature(graph):
    """Return edge-curvature summaries and edge artifacts without mutating the graph."""
    from GraphRicciCurvature.FormanRicci import FormanRicci

    union = graph.undirected.copy()
    for _, data in union.nodes(data=True):
        data["weight"] = 1.0
    for _, _, data in union.edges(data=True):
        data["weight"] = 1.0
    if not union.number_of_edges():
        raise ValueError("Forman curvature requires at least one edge")
    model = FormanRicci(union, method="1d", verbose="ERROR")
    model.compute_ricci_curvature()
    edges = {(i, j): float(data["formanCurvature"])
             for i, j, data in model.G.edges(data=True)}
    values = np.asarray(list(edges.values()))
    metrics = dict(mean=values.mean(), median=np.median(values), std=values.std(),
                   q10=np.quantile(values, .1), q90=np.quantile(values, .9),
                   negative_fraction=np.mean(values < 0))
    return {"forman_" + key: value for key, value in metrics.items()}, {}, {"forman_curvature": edges}
