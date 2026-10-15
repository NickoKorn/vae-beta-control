"""Descriptive X/mu differences, sklearn ARI/NMI, SciPy hub Spearman.

Directed edge overlap is not part of this active comparison. No cancer labels
are used; zero delta does not imply optimality.
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def compare_graphs(input_graph, latent_graph, input_metrics, latent_metrics, input_nodes, latent_nodes):
    if input_graph.k != latent_graph.k or not np.array_equal(input_graph.sample_ids, latent_graph.sample_ids):
        raise ValueError("Graph comparison requires identical ordered IDs and k")
    out = {"delta_"+key: latent_metrics[key]-input_metrics[key] for key in sorted(input_metrics.keys() & latent_metrics.keys())}
    out.update(partition_ari=np.nan, partition_nmi=np.nan, hub_identity_spearman=np.nan)
    if "community" in input_nodes and "community" in latent_nodes:
        a, b = input_nodes["community"], latent_nodes["community"]
        out.update(partition_ari=adjusted_rand_score(a, b), partition_nmi=normalized_mutual_info_score(a, b))
    if "k_occurrence" in input_nodes and "k_occurrence" in latent_nodes:
        out["hub_identity_spearman"] = spearmanr(input_nodes["k_occurrence"], latent_nodes["k_occurrence"]).statistic
    return out
