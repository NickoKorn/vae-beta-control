"""Core information-theoretic posterior metrics."""

from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..metric_pipeline import CoreMetricInputs

import numpy as np
import pandas as pd
import torch
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence


def compute_conditional_kl_matrix(
    mu: np.ndarray,
    logvar: np.ndarray,
) -> np.ndarray:
    """Return conditional Gaussian-to-prior KL per sample and dimension.

    Input representation:
        Aligned posterior ``mu`` and ``logvar`` matrices from one VARIX split.

    The output has shape ``(n_samples, latent_dim)``. It is the shared
    mathematical input for rate aggregation and epsilon-delta collapse, not a
    reported metric by itself. The calculation delegates to PyTorch's
    ``kl_divergence`` for diagonal Normal posteriors.
    """
    means = np.asarray(mu, dtype=np.float64)
    log_variances = np.asarray(logvar, dtype=np.float64)
    if means.ndim != 2 or means.shape != log_variances.shape:
        raise ValueError("mu and logvar must be equal-shape two-dimensional arrays")
    if means.shape[0] == 0 or means.shape[1] == 0:
        raise ValueError("mu and logvar must have non-zero sample and latent axes")
    if not np.isfinite(means).all() or not np.isfinite(log_variances).all():
        raise ValueError("mu and logvar must contain only finite values")

    mu_t = torch.as_tensor(means, dtype=torch.float64)
    logvar_t = torch.as_tensor(log_variances, dtype=torch.float64)

    # logvar = log(sigma^2), so the posterior standard deviation is exp(0.5*logvar).
    posterior = Normal(mu_t, torch.exp(0.5 * logvar_t))
    prior = Normal(torch.zeros_like(mu_t), torch.ones_like(mu_t))

    return kl_divergence(posterior, prior).cpu().numpy()


def compute_rate(conditional_kl: np.ndarray) -> pd.DataFrame:
    """Report the empirical VAE rate in nats per sample.

    Input representation:
        The sample-by-dimension matrix returned by
        :func:`compute_conditional_kl_matrix`.

    Mathematical definition:
        For each sample, sum the conditional KL over latent dimensions and then
        average across samples:

            R = E_x[KL(q(z|x) || p(z))].

    This mean conditional KL is the reported VAE rate. No median or quantile
    summaries are promoted to metrics here.
    """
    matrix = np.asarray(conditional_kl, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("conditional_kl must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("conditional_kl must contain only finite values")

    rate_per_sample = np.sum(matrix, axis=1)

    return pd.DataFrame.from_records([
        {
            "metric_name": "posterior_conditional_kl_total_mean",
            "value": float(np.mean(rate_per_sample)),
            "status": "ok",
            "warning": None,
            "n_samples": len(rate_per_sample),
            "split": "test",
        }
    ])


def compute_autoencodix_total_correlation(varix: Any, z: np.ndarray) -> pd.DataFrame:
    """Compute exact AUTOENCODIX Gaussian covariance total correlation.

    Input representation:
        Persisted sampled latent draws ``z_test`` from AUTOENCODIX
        ``result.latentspaces``. This empirical sampled aggregated-posterior
        representation is not posterior mean ``mu``.

    Metric family:
        Information theory.
    Space comparison:
        Aggregated posterior only.
    Exact spaces:
        Joint sampled AUTOENCODIX latent coordinates versus their Gaussian marginals.
    Representations:
        Sampled latent coordinates z from result.latentspaces.
    Mathematical definition:
        TC=0.5*(sum_i log Sigma_ii-log det Sigma).
    Scientific source:
        Joas et al., AUTOENCODIX; Rowe and Day, Entropy 2019, 21, 921.
    Repository provenance:
        Direct call to ``varix.visualizer._total_correlation`` on the visualizer
        instance owned by the trained VARIX pipeline.
    Interpretation:
        Gaussian covariance dependence among latent coordinates; zero means
        uncorrelated coordinates and implies independence only under Gaussianity.
    Limitations:
        Captures covariance dependence only and is unstable for singular covariance.
    Direction:
        Lower is better for coordinate independence.
    In core metrics included:
        Yes, as a sampled-z AUTOENCODIX TC candidate compared separately with
        a potential beta-TCVAE mixture estimator (not active in this pipeline).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        tc = float(varix.visualizer._total_correlation(pd.DataFrame(z)))
    if not np.isfinite(tc):
        warning = (
            "AUTOENCODIX total correlation is undefined because the latent "
            "covariance matrix is singular or contains zero-variance dimensions."
        )
        return pd.DataFrame.from_records([
            {
                "metric_name": "info_total_correlation_autoencodix",
                "value": float("nan"),
                "status": "degenerate_input",
                "warning": warning,
                "n_samples": len(z),
                "split": "test",
            }
        ])
    return pd.DataFrame.from_records([
        {
            "metric_name": "info_total_correlation_autoencodix",
            "value": tc,
            "status": "ok",
            "warning": None,
            "n_samples": len(z),
            "split": "test",
        }
    ])


def run_information_theoretic_metrics_pipeline(
    inputs: CoreMetricInputs,
    varix: Any,
    *, include_total_correlation: bool = True,
) -> pd.DataFrame:
    """Run core conditional rate and sampled-z TC diagnostics.

    Input representation:
        ``mu_test`` and ``logvar_test`` parameterize each diagonal conditional
        posterior for KL/rate; stored ``z_test`` alone is passed to the
        AUTOENCODIX covariance total-correlation routine. These inputs are not
        interchangeable.

    This is orchestration, not a metric and not a representation-quality score.

    In core metrics included: Yes, as the information group runner.
    """
    frames: list[pd.DataFrame] = []

    if include_total_correlation and inputs.z_test is not None:
        frames.append(compute_autoencodix_total_correlation(varix, inputs.z_test))

    if inputs.logvar_test is None:
        return (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(
                columns=("metric_name", "value", "status", "warning", "n_samples", "split")
            )
        )

    frames.append(
        compute_rate(
            compute_conditional_kl_matrix(
                inputs.mu_test,
                inputs.logvar_test,
            )
        )
    )

    return pd.concat(frames, ignore_index=True)


__all__ = [
    "compute_autoencodix_total_correlation",
    "compute_conditional_kl_matrix",
    "compute_rate",
    "run_information_theoretic_metrics_pipeline",
]
