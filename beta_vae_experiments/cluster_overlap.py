"""Label-free cluster overlap between the input X and a latent embedding.

Reference clusters come from k-means on X; their number is the one with the best
silhouette (ZADU ``internal_validation_measure``). For a latent embedding we then ask
three questions without any annotation labels:

* How many clusters does the latent space itself support (same silhouette rule)?
* Are the X clusters recovered there (ZADU ``clustering_and_external_validation_measure``,
  k-means with as many clusters as X has, adjusted Rand index)?
* How much do the X clusters overlap in the latent neighbourhoods? Same-cluster share of
  the k nearest neighbours (ZADU ``neighborhood_hit``, self-first neighbour lists so that
  duplicate points at mu = 0 count correctly), split into a cluster-by-cluster matrix.

``describe`` turns one result into plain sentences, so a user or an agent can read the
overlap without looking at a plot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from zadu.measures import clustering_and_external_validation_measure, internal_validation_measure, neighborhood_hit

from beta_vae_experiments.preservation_annealing import _self_first_neighbours

CANDIDATE_CLUSTERS = tuple(range(2, 21))


def kmeans_labels(values, n_clusters, seed=0):
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(values)


def best_clustering(values, candidates=CANDIDATE_CLUSTERS, seed=0):
    """k-means labels with the number of clusters that maximises the silhouette."""
    scores, labels = {}, {}
    for n_clusters in candidates:
        if len(np.unique(values, axis=0)) <= n_clusters:
            break
        labels[n_clusters] = kmeans_labels(values, n_clusters, seed)
        scores[n_clusters] = internal_validation_measure.measure(values, labels[n_clusters], "silhouette")["silhouette"]
    best = max(scores, key=scores.get)
    return labels[best], pd.Series(scores, name="silhouette")


def neighbour_matrix(neighbours, labels):
    """Share of the k nearest neighbours of cluster a that belong to cluster b (rows sum to 1)."""
    clusters = np.unique(labels)
    counts = pd.crosstab(np.repeat(labels, neighbours.shape[1]), labels[neighbours].ravel()).reindex(
        index=clusters, columns=clusters, fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0)


def pair_overlap(matrix, sizes):
    """Symmetric overlap of each cluster pair: neighbour relations between a and b / all relations of a and b."""
    rows = []
    for i, a in enumerate(matrix.index):
        for b in matrix.index[i + 1:]:
            shared = matrix.loc[a, b] * sizes[a] + matrix.loc[b, a] * sizes[b]
            rows.append(dict(a=a, b=b, overlap=shared / (sizes[a] + sizes[b])))
    return pd.DataFrame(rows)


class ClusterReference:
    """Reference clustering of X, computed once per run."""

    def __init__(self, x, k=20, seed=0, labels=None, silhouettes=None):
        """``labels``/``silhouettes`` restore a cached clustering of the same X instead of recomputing it."""
        self.k = k
        if labels is None:
            labels, silhouettes = best_clustering(x, seed=seed)
        self.labels, self.silhouettes = np.asarray(labels), pd.Series(silhouettes, name="silhouette")
        self.n_clusters = int(len(np.unique(self.labels)))
        self.sizes = pd.Series(self.labels).value_counts().sort_index()
        self.chance = float(((self.sizes / self.sizes.sum()) ** 2).sum())
        neighbours, _ = _self_first_neighbours(np.asarray(x, dtype=np.float64), k)
        self.hit = neighborhood_hit.measure(x, self.labels, k=k, knn_emb_info=neighbours)["neighborhood_hit"]
        self.matrix = neighbour_matrix(neighbours, self.labels)
        self.pairs = pair_overlap(self.matrix, self.sizes)

    def same_cluster_share(self, latent):
        """Only the neighbourhood-hit part of ``evaluate`` (cheap, e.g. for the noisy z)."""
        neighbours, _ = _self_first_neighbours(np.asarray(latent, dtype=np.float64), self.k)
        return neighborhood_hit.measure(latent, self.labels, k=self.k, knn_emb_info=neighbours)["neighborhood_hit"]

    def evaluate(self, latent, seed=0):
        """Overlap of the X clusters in a latent embedding of the same patients (same order)."""
        latent = np.asarray(latent, dtype=np.float64)
        neighbours, _ = _self_first_neighbours(latent, self.k)
        own_labels, own_silhouettes = best_clustering(latent, seed=seed)
        matrix = neighbour_matrix(neighbours, self.labels)
        pairs = pair_overlap(matrix, self.sizes).merge(self.pairs, on=["a", "b"], suffixes=("", "_x"))
        pairs["increase"] = pairs.overlap - pairs.overlap_x
        own_cluster = pd.Series(np.diag(matrix), index=matrix.index)
        return dict(
            n_clusters_x=self.n_clusters, silhouette_x=float(self.silhouettes.max()),
            n_clusters_latent=int(len(np.unique(own_labels))), silhouette_latent=float(own_silhouettes.max()),
            silhouette_x_clusters_in_latent=internal_validation_measure.measure(latent, self.labels, "silhouette")["silhouette"],
            recovery_ari=clustering_and_external_validation_measure.measure(
                latent, self.labels, "arand", "kmeans",
                dict(n_clusters=self.n_clusters, n_init=10, random_state=seed))["kmeans_arand"],
            same_cluster_neighbours_x=float(self.hit),
            same_cluster_neighbours_latent=neighborhood_hit.measure(latent, self.labels, k=self.k,
                                                                     knn_emb_info=neighbours)["neighborhood_hit"],
            chance=self.chance, k=self.k,
            own_cluster_share=own_cluster, own_cluster_share_x=pd.Series(np.diag(self.matrix), index=matrix.index),
            pairs=pairs.sort_values("increase", ascending=False).reset_index(drop=True))


def scalar_summary(result):
    """The numeric part of ``ClusterReference.evaluate`` as one flat dict (for CSV rows)."""
    top = result["pairs"].iloc[0]
    return {key: value for key, value in result.items() if np.isscalar(value)} | dict(
        top_pair=f"{top.a}–{top.b}", top_pair_overlap=top.overlap, top_pair_overlap_x=top.overlap_x)


def _pct(value):
    return f"{100 * value:.0f} %"


def _num(value):
    return f"{value:.2f}".replace(".", ",")


def _strength(silhouette):
    return "kaum Clusterstruktur" if silhouette < .1 else "schwache Clusterstruktur" if silhouette < .25 else \
        "mittlere Clusterstruktur" if silhouette < .5 else "deutliche Clusterstruktur"


def describe(result, cluster_names=None, n_pairs=3):
    """Plain-language overlap report for one embedding."""
    name = (lambda c: f"{c} ({cluster_names[c]})") if cluster_names else str
    lines = [
        f"X hat {result['n_clusters_x']} Cluster (k-means, beste Silhouette {_num(result['silhouette_x'])}: "
        f"{_strength(result['silhouette_x'])}); der Latentraum selbst trägt {result['n_clusters_latent']} Cluster "
        f"(Silhouette {_num(result['silhouette_latent'])}: {_strength(result['silhouette_latent'])}).",
        f"Von den {result['k']} nächsten Nachbarn eines Patienten liegen im Latentraum "
        f"{_pct(result['same_cluster_neighbours_latent'])} im eigenen X-Cluster (in X: {_pct(result['same_cluster_neighbours_x'])}, "
        f"Zufall: {_pct(result['chance'])}). Der Overlap ist also "
        f"{_pct(1 - result['same_cluster_neighbours_latent'])} statt {_pct(1 - result['same_cluster_neighbours_x'])}.",
        f"Die X-Cluster sind im Latentraum mit ARI {_num(result['recovery_ari'])} wiederzufinden "
        f"(1 = identisch, 0 = Zufall); Silhouette der X-Cluster im Latentraum {_num(result['silhouette_x_clusters_in_latent'])}.",
    ]
    pairs = result["pairs"].head(n_pairs)
    if len(pairs) and pairs.increase.iloc[0] > 0.01:
        merged = "; ".join(f"{name(p.a)} ↔ {name(p.b)}: {_pct(p.overlap)} gemeinsame Nachbarn (X: {_pct(p.overlap_x)})"
                           for p in pairs.itertuples() if p.increase > 0.01)
        lines.append(f"Am stärksten zusammengewachsen: {merged}.")
    weakest = (result["own_cluster_share"] - result["own_cluster_share_x"]).sort_values().head(2)
    lines.append("Am stärksten aufgelöst: " + "; ".join(
        f"{name(c)} behält {_pct(result['own_cluster_share'][c])} Nachbarn im eigenen Cluster "
        f"(X: {_pct(result['own_cluster_share_x'][c])})" for c in weakest.index) + ".")
    return "\n".join(lines)
