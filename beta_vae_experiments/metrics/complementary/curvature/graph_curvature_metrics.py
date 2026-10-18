"""GraphRicciCurvature Forman (1d) geometry of the unit-weight union graph.

Representation distances remain separate from curvature weights. Negative
curvature may describe bottleneck-like edges; utility here is an empirical
question. Optional Ollivier uses package defaults, one process, alpha=.5.
See graph_metrics.md and the GraphRicciCurvature software documentation.
"""
import numpy as np
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci


def compute_curvature(graph, ollivier=False):
    g = graph.undirected.copy()
    if ollivier:
        model = OllivierRicci(g, alpha=.5, proc=1, verbose="ERROR")
        attribute, prefix = "ricciCurvature", "ollivier"
    else:
        model = FormanRicci(g, method="1d", verbose="ERROR")
        attribute, prefix = "formanCurvature", "forman"
    model.compute_ricci_curvature()
    edges = {(i, j): float(data[attribute]) for i, j, data in model.G.edges(data=True)}
    values = np.asarray(list(edges.values()))
    metrics = dict(mean=values.mean(), median=np.median(values), std=values.std(),
                   q10=np.quantile(values, .1), q90=np.quantile(values, .9),
                   negative_fraction=np.mean(values < 0))
    return {prefix+"_"+key: val for key, val in metrics.items()}, {}, {prefix+"_curvature": edges}
