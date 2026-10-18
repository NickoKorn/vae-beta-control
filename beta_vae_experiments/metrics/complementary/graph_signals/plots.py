"""Two small plots for the graph-frequency analysis in ``graph_spectrum.py``."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_dim_band_heatmap(band_energy: np.ndarray, dim_labels: Sequence[str] | None = None, band_labels: Sequence[str] | None = None, ax=None):
    """Heatmap latent dimension × frequency band, band_energy (d, n_bands) as shares."""
    ax = ax or plt.subplots(figsize=(1.0 + 0.7 * band_energy.shape[1], 1.0 + 0.4 * band_energy.shape[0]))[1]
    im = ax.imshow(band_energy, aspect="auto", cmap="viridis", vmin=0, vmax=max(float(np.nanmax(band_energy)), 1e-12))
    ax.set_yticks(range(band_energy.shape[0]), dim_labels or [f"z{j + 1}" for j in range(band_energy.shape[0])])
    ax.set_xticks(range(band_energy.shape[1]), band_labels or [f"band {b}" for b in range(band_energy.shape[1])], rotation=45)
    ax.set_xlabel("frequency band (low → high)")
    ax.figure.colorbar(im, ax=ax, label="energy share")
    return ax


def plot_metric_condition_heatmap(values: np.ndarray, metric_labels: Sequence[str], condition_labels: Sequence[str], ax=None):
    """Heatmap metric × filter condition, values (n_metrics, n_conditions) relative to unfiltered (1 = unchanged)."""
    ax = ax or plt.subplots(figsize=(1.0 + 0.5 * values.shape[1], 1.0 + 0.45 * values.shape[0]))[1]
    span = max(float(np.nanmax(np.abs(values - 1.0))), 1e-12)
    im = ax.imshow(values, aspect="auto", cmap="RdBu", vmin=1 - span, vmax=1 + span)
    ax.set_yticks(range(values.shape[0]), metric_labels)
    ax.set_xticks(range(values.shape[1]), condition_labels, rotation=90)
    ax.figure.colorbar(im, ax=ax, label="metric / unfiltered")
    return ax
