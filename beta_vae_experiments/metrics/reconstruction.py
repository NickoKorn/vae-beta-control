"""Reconstruction core metrics."""

from __future__ import annotations

import numpy as np

from .contracts import (
    CoreMetricInputs,
    MetricGroupResult,
    MetricResult,
    MetricValue,
    ReconstructionMetricConfig,
)


def compute_reconstruction_mse(
    x: np.ndarray, x_hat: np.ndarray, split: str
) -> MetricResult:
    """Compute mean squared reconstruction error.

    Metric family:
        Reconstruction.
    Space comparison:
        Input versus reconstruction.
    Exact spaces:
        Aligned preprocessed model-input and decoded reconstruction matrices for one split.
    Representations:
        X and X_hat.
    Mathematical definition:
        MSE = (1/(n*d)) sum_ij (X_ij-X_hat_ij)^2.
    Scientific source:
        Kingma and Welling, Auto-Encoding Variational Bayes, arXiv:1312.6114.
    Repository provenance:
        NumPy implementation of the cited definition.
    Interpretation:
        Average squared reconstruction error.
    Limitations:
        Scale-dependent and not comparable across incompatible preprocessing.
    Direction:
        Lower is better.
    """
    value = float(np.mean(np.square(x - x_hat), dtype=np.float64))
    return MetricResult(
        [MetricValue(f"recon_{split}_mse", value, n_samples=len(x), split=split)]
    )


def compute_reconstruction_explained_variance(
    x: np.ndarray, x_hat: np.ndarray, split: str
) -> MetricResult:
    """Compute AUTOENCODIX-style mean per-sample explained variance.

    pyDRMetrics compatibility note:
        This is an AUTOENCODIX/scikit-learn-style per-sample explained-variance
        score. It is neither pyDRMetrics reconstruction MSE nor its relative
        reconstruction error ``MSE(X, X_hat) / MSE(X, 0)``.

    Metric family:
        Reconstruction.
    Space comparison:
        Input versus reconstruction.
    Exact spaces:
        Feature vectors of aligned observations in X_split and X_hat_split.
    Representations:
        X and X_hat.
    Mathematical definition:
        For each sample, 1-Var(X-X_hat)/Var(X), then arithmetic mean over samples.
    Scientific source:
        Joas et al., AUTOENCODIX, DOI:10.1038/s43588-025-00916-4.
    Repository provenance:
        sklearn explained_variance_score convention applied per observation.
    Interpretation:
        Mean fraction of within-observation feature variance explained by reconstruction.
    Limitations:
        Undefined for constant feature vectors and sensitive to preprocessing.
    Direction:
        Higher is better.
    """
    residual_variance = np.var(x - x_hat, axis=1, dtype=np.float64)
    target_variance = np.var(x, axis=1, dtype=np.float64)
    scores = np.full(len(x), np.nan, dtype=np.float64)
    valid = target_variance > np.finfo(np.float64).eps
    scores[valid] = 1.0 - residual_variance[valid] / target_variance[valid]
    warning = None
    status = "ok"
    if not np.any(valid):
        status = "degenerate_input"
        warning = "All observations have zero feature variance."
    elif not np.all(valid):
        warning = f"{int(np.sum(~valid))} constant observations were excluded."
    value = float(np.nanmean(scores)) if np.any(valid) else float("nan")
    return MetricResult(
        [
            MetricValue(
                f"recon_{split}_explained_variance_mean",
                value,
                status=status,
                warning=warning,
                n_samples=len(x),
                split=split,
            )
        ],
        warnings=[warning] if warning else [],
    )


def compute_reconstruction_generalization_gap(
    train_mse: float,
    val_mse: float,
    epsilon: float,
) -> MetricResult:
    """Compute validation-minus-training MSE gap and validation/train ratio.

    Metric family:
        Reconstruction.
    Space comparison:
        Cross-split reconstruction.
    Exact spaces:
        Training input/reconstruction errors and validation input/reconstruction errors.
    Representations:
        X_train, X_hat_train, X_val, and X_hat_val.
    Mathematical definition:
        Gap=MSE_val-MSE_train; ratio=MSE_val/max(MSE_train, epsilon).
    Scientific source:
        Derived diagnostic based on the VAE reconstruction term of Kingma and Welling.
    Repository provenance:
        Project implementation from the stated equations.
    Interpretation:
        Measures reconstruction generalization from training to validation data.
    Limitations:
        These are derived diagnostics, not independent canonical VAE objectives.
    Direction:
        Lower is generally better.
    """
    return MetricResult(
        [
            MetricValue("recon_val_train_gap", float(val_mse - train_mse)),
            MetricValue(
                "recon_val_train_ratio",
                float(val_mse / max(train_mse, epsilon)),
            ),
        ]
    )


def run_reconstruction_metrics_pipeline(
    inputs: CoreMetricInputs,
    config: ReconstructionMetricConfig,
) -> MetricGroupResult:
    results: list[MetricResult] = []
    for split in ("train", "val", "test"):
        x = getattr(inputs, f"x_{split}")
        x_hat = getattr(inputs, f"x_recon_{split}")
        results.append(compute_reconstruction_mse(x, x_hat, split))
        results.append(compute_reconstruction_explained_variance(x, x_hat, split))
    mse = {
        value.name: float(value.value)
        for result in results
        for value in result.values
        if value.name.endswith("_mse")
    }
    results.append(
        compute_reconstruction_generalization_gap(
            mse["recon_train_mse"], mse["recon_val_mse"], config.epsilon
        )
    )
    values = [value for result in results for value in result.values]
    return MetricGroupResult(
        group_name="reconstruction",
        status=(
            "ok"
            if all(value.status == "ok" for value in values)
            else "degenerate_input"
        ),
        values=values,
        warnings=[warning for result in results for warning in result.warnings],
    )
