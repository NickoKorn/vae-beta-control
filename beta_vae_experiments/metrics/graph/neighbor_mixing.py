"""Group mixing on directed kNN neighborhoods of mu, z or posterior-W2 embeddings.

This builds neighbors on the eligible cohort, excluding the query itself.
It does not use the symmetrized union graph of the structural graph metrics.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def foreign_neighbor_fraction(embedding, reference_groups, k=20):
    """Local mixing using sklearn neighbors, excluding self and missing/-1 groups.

    Return raw fractions and fractions divided by the finite-sample random-mixing
    expectation (N - own_group_size)/(N - 1). Normalized 1 is random mixing,
    not a maximum. Identical point clouds have undefined rank neighborhoods.
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
    """
    values = np.asarray(embedding, dtype=float)
    groups = np.asarray(reference_groups)
    if values.ndim != 2 or len(groups) != len(values) or not np.isfinite(values).all():
        raise ValueError('Embedding must be finite and align with group labels')
    keep = pd.notna(groups) & (groups != -1)
    values, groups = values[keep], groups[keep]
    if not 1 <= k < len(values):
        raise ValueError('Require 1 <= k < number of eligible patients')
    if len(pd.unique(groups)) < 2:
        raise ValueError('Need at least two groups to measure mixing')
    if np.all(values == values[0]):
        raise ValueError('All points are identical; nearest-neighbor ranks are undefined')
    neighbors = NearestNeighbors(n_neighbors=k).fit(values).kneighbors(return_distance=False)
    fractions = (groups[neighbors] != groups[:, None]).mean(axis=1)
    counts = pd.Series(groups).value_counts()
    expected = (len(groups) - counts.reindex(groups).to_numpy()) / (len(groups) - 1)
    forced_floor = np.maximum(0., 1 - (counts.reindex(groups).to_numpy() - 1) / k)
    return dict(patient_mask=keep, groups=groups, neighbor_indices=neighbors,
                per_patient=fractions, random_mixing_expectation=expected,
                forced_foreign_floor=forced_floor,
                normalized=fractions / expected,
                duplicate_fraction=1 - len(np.unique(values, axis=0)) / len(values))
