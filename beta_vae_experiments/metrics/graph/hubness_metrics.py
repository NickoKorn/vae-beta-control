"""scikit-hubness diagnostics on the canonical sparse kNN graph.

Uses 0.30.0a1 Hubness.fit/score, without neighbor reshuffling or reduction.
Hub occurrence is edge mass; hub fraction is fraction of library-identified
nodes. These differ.

Radovanović et al. (2010): https://jmlr.org/papers/v11/radovanovic10a.html
Biologische Anwendung: https://academic.oup.com/bioinformatics/article/38/4/1045/6433673
Die Quellen belegen keine Vorhersage von TCGA-VAE-AUC-Verlusten. Weniger
Hubness ist nicht universell besser. Tie-Auflösung nicht still randomisieren.
"""


def compute_hubness(graph, seed=31):
    from skhubness import Hubness
    model = Hubness(k=graph.k, metric="precomputed", shuffle_equal=False,
                    return_value="all", return_k_occurrence=True,
                    return_hubs=True, return_antihubs=True, random_state=seed)
    scores = model.fit(graph.neighbors).score(graph.neighbors)
    counts = scores["k_occurrence"]
    metrics = {"hubness_skewness": scores["k_skewness"],
               "antihub_fraction": scores["antihub_occurrence"],
               "hub_occurrence": scores["hub_occurrence"],
               "hub_fraction": len(scores["hubs"])/len(counts),
               "robinhood": scores["robinhood"], "groupie_ratio": scores["groupie_ratio"]}
    return metrics, {"k_occurrence": counts}
