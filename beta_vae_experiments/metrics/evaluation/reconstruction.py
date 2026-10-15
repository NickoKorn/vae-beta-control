"""Reconstruction core metrics."""

from __future__ import annotations

import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...metric_pipeline import CoreMetricInputs

import numpy as np
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score


def compute_reconstruction_mse(
    x: np.ndarray, x_hat: np.ndarray, split: str
) -> pd.DataFrame:
    """Compute mean squared reconstruction error.

    Input representation:
        Row- and feature-aligned preprocessed input ``x`` and decoded
        reconstruction ``x_hat`` from the named split; no latent representation.

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
    In core metrics included: Yes
    """
    mse = float(mean_squared_error(x, x_hat))
    return pd.DataFrame.from_records([{"metric_name": f"recon_{split}_mse", "value": mse, "status": "ok", "warning": None, "n_samples": len(x), "split": split}])


def compute_reconstruction_r2(
    x: np.ndarray, x_hat: np.ndarray, split: str
) -> pd.DataFrame:
    """Compute the standard uniformly averaged multi-output R² score.

    Input representation:
        Aligned preprocessed inputs and reconstructions for one data split.
    Metric family:
        Reconstruction quality.
    Mathematical definition:
        Feature-wise coefficient of determination, uniformly averaged across
        features by ``sklearn.metrics.r2_score``.
    Scientific source:
        Standard coefficient of determination; operational definition follows
        the installed scikit-learn implementation.
    Repository provenance:
        Direct library call with ``multioutput='uniform_average'`` and
        ``force_finite=True``.
    Interpretation:
        Fraction of feature-wise variation captured by reconstruction, with
        systematic reconstruction offsets penalized.
    Limitations:
        Can be negative and depends on feature preprocessing. Constant-feature
        cases use scikit-learn's finite replacement convention.
    Direction:
        Higher is better.
    In core metrics included:
        Yes, as confirmatory reconstruction evidence.
    """
    value = float(
        r2_score(
            x,
            x_hat,
            multioutput="uniform_average",
            force_finite=True,
        )
    )
    return pd.DataFrame.from_records([
            {"metric_name": f"recon_{split}_r2_uniform_average", "value": value, "status": "ok", "warning": None, "n_samples": len(x), "split": split}
        ])


def compute_reconstruction_explained_variance(
    x: np.ndarray, x_hat: np.ndarray, split: str
) -> pd.DataFrame:
    """Compute scikit-learn's uniformly averaged explained-variance score.

    Input representation:
        Aligned preprocessed inputs and reconstructions for one data split.
    Metric family:
        Reconstruction quality.
    Mathematical definition:
        Feature-wise ``1-Var(X-X_hat)/Var(X)``, uniformly averaged across
        features by ``sklearn.metrics.explained_variance_score``.
    Scientific source:
        Standard explained-variance score; operational definition follows the
        installed scikit-learn implementation.
    Repository provenance:
        Direct library call with ``multioutput='uniform_average'`` and
        ``force_finite=True``.
    Interpretation:
        Measures preservation of variation while ignoring constant residual
        offsets, unlike R².
    Limitations:
        Depends on feature preprocessing and can mask systematic offsets.
        Constant-feature cases use scikit-learn's finite replacement rule.
    Direction:
        Higher is better.
    In core metrics included:
        Yes, as confirmatory reconstruction evidence.
    """
    value = float(
        explained_variance_score(
            x,
            x_hat,
            multioutput="uniform_average",
            force_finite=True,
        )
    )
    return pd.DataFrame.from_records([
            {"metric_name": f"recon_{split}_explained_variance_uniform_average", "value": value, "status": "ok", "warning": None, "n_samples": len(x), "split": split}
        ])


def compute_reconstruction_generalization_gap(inputs: CoreMetricInputs) -> pd.DataFrame:
    """Return validation MSE minus training MSE as generalization QC.

    Input representation: aligned train/validation inputs and reconstructions.
    Metric family: reconstruction generalization.
    Mathematical definition: ``MSE_val-MSE_train`` using sklearn MSE on each split.
    Scientific source: standard empirical generalization-gap diagnostic.
    Repository provenance: project difference of two library-backed MSE values.
    Interpretation: positive values indicate worse held-out validation reconstruction.
    Limitations: split size/noise and distribution shift affect the value; it is not latent quality.
    Direction: values closer to zero usually indicate similar split fit.
    In core metrics included: QC only.
    """
    train_mse = float(mean_squared_error(inputs.x_train, inputs.x_recon_train))
    val_mse = float(mean_squared_error(inputs.x_val, inputs.x_recon_val))
    return pd.DataFrame.from_records([{"metric_name": "recon_generalization_gap", "value": val_mse - train_mse, "status": "ok", "warning": None, "n_samples": None, "split": "val_minus_train"}])


def run_reconstruction_metrics_pipeline(
    inputs: CoreMetricInputs,
) -> pd.DataFrame:
    """Run selected reconstruction-utility metrics on train/validation/test.

    Input representation:
        Aligned ``x_{train,val,test}`` and ``x_recon_{train,val,test}`` matrices.
        Posterior means, posterior variances, and sampled latents are not used.

    The pipeline only orchestrates independently registered leaf metrics and
    does not combine MSE and R² into a score. All arrays are assumed to have
    passed the shared input and sample-alignment validation.

    In core metrics included: Yes, as the reconstruction group runner.
    """
    frames: list[pd.DataFrame] = []
    for split in ("train", "val", "test"):
        x = getattr(inputs, f"x_{split}")
        x_hat = getattr(inputs, f"x_recon_{split}")
        frames.append(compute_reconstruction_mse(x, x_hat, split))
        frames.append(compute_reconstruction_r2(x, x_hat, split))
        frames.append(compute_reconstruction_explained_variance(x, x_hat, split))
    frames.append(compute_reconstruction_generalization_gap(inputs))
    return pd.concat(frames, ignore_index=True)
