"""Scientifically distinct information-theoretic estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .contracts import (
    CoreMetricInputs,
    InformationMetricConfig,
    MetricGroupResult,
    MetricResult,
    MetricValue,
)


def _normal_log_density(
    values: np.ndarray, means: np.ndarray, logvars: np.ndarray
) -> np.ndarray:
    return -0.5 * (
        np.log(2.0 * np.pi) + logvars + np.square(values - means) / np.exp(logvars)
    )


def _select_posterior_samples(
    mu: np.ndarray,
    logvar: np.ndarray,
    z: np.ndarray | None,
    config: InformationMetricConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    n = len(mu)
    if n > config.max_samples:
        rng = np.random.default_rng(config.random_state)
        indices = np.sort(rng.choice(n, config.max_samples, replace=False))
    else:
        indices = np.arange(n)
    mus = np.asarray(mu[indices], dtype=np.float64)
    logvars = np.clip(np.asarray(logvar[indices], dtype=np.float64), -30.0, 30.0)
    if z is not None:
        samples = np.asarray(z[indices], dtype=np.float64)
        source = "stored_z"
    else:
        rng = np.random.default_rng(config.random_state)
        samples = mus + rng.normal(size=mus.shape) * np.exp(0.5 * logvars)
        source = "deterministic_resample"
    return mus, logvars, samples, source


def _estimate_beta_tcvae_components(
    mu: np.ndarray,
    logvar: np.ndarray,
    z: np.ndarray | None,
    config: InformationMetricConfig,
) -> tuple[float, float, float, int, str]:
    mus, logvars, samples, source = _select_posterior_samples(mu, logvar, z, config)
    n = len(mus)
    log_n = np.log(float(n))
    mi_parts: list[np.ndarray] = []
    tc_parts: list[np.ndarray] = []
    dw_parts: list[np.ndarray] = []
    for start in range(0, n, config.chunk_size):
        stop = min(n, start + config.chunk_size)
        chunk = samples[start:stop]
        component_dim = _normal_log_density(
            chunk[:, None, :], mus[None, :, :], logvars[None, :, :]
        )
        component_joint = np.sum(component_dim, axis=2)
        log_qz = logsumexp(component_joint, axis=1) - log_n
        log_qz_product = np.sum(logsumexp(component_dim, axis=1) - log_n, axis=1)
        own_log_qzx = np.sum(
            _normal_log_density(chunk, mus[start:stop], logvars[start:stop]), axis=1
        )
        log_pz = np.sum(-0.5 * (np.log(2.0 * np.pi) + np.square(chunk)), axis=1)
        mi_parts.append(own_log_qzx - log_qz)
        tc_parts.append(log_qz - log_qz_product)
        dw_parts.append(log_qz_product - log_pz)
    return (
        float(np.mean(np.concatenate(mi_parts))),
        float(np.mean(np.concatenate(tc_parts))),
        float(np.mean(np.concatenate(dw_parts))),
        n,
        source,
    )


def estimate_beta_tcvae_mixture_information_components(
    mu: np.ndarray,
    logvar: np.ndarray,
    z: np.ndarray | None,
    config: InformationMetricConfig,
) -> MetricResult:
    """Estimate the three beta-TCVAE KL components with one mixture estimator.

    Metric family:
        Information theory.
    Space comparison:
        Input identity, aggregated posterior, posterior marginals, and prior.
    Exact spaces:
        Conditional q(z|x), aggregated q(z), product of aggregated marginals,
        and standard-normal p(z).
    Representations:
        Posterior parameters and stored or deterministically resampled z.
    Mathematical definition:
        MI=E[log q(z|x)-log q(z)], TC=E[log q(z)-sum_j log q(z_j)],
        and DWKL=E[sum_j log q(z_j)-log p(z)].
    Scientific source:
        Chen et al., Isolating Sources of Disentanglement, NeurIPS 2018.
    Repository provenance:
        Project beta-TCVAE-style finite-mixture density estimator.
    Interpretation:
        Separates retained sample information, non-factorial dependence, and
        marginal prior mismatch using one internally consistent estimator.
    Limitations:
        Finite-sample mixture estimates are biased and scale quadratically in
        the selected sample count; the TC value is distinct from AUTOENCODIX TC.
    Direction:
        Descriptive; lower TC indicates a more factorized aggregated posterior.
    """
    mi, tc, dwkl, n, source = _estimate_beta_tcvae_components(mu, logvar, z, config)
    selected_mu, selected_logvar, _, _ = _select_posterior_samples(
        mu, logvar, z, config
    )
    rate_reference = float(
        np.mean(
            np.sum(
                0.5
                * (
                    np.square(selected_mu)
                    + np.exp(selected_logvar)
                    - selected_logvar
                    - 1.0
                ),
                axis=1,
            )
        )
    )
    decomposition_residual = rate_reference - (mi + tc + dwkl)
    estimator = "beta_tcvae_mixture_density"
    return MetricResult(
        [
            MetricValue("info_mutual_information_nats", mi, n_samples=n, split="test"),
            MetricValue(
                "info_total_correlation_beta_tcvae_mixture_nats",
                tc,
                n_samples=n,
                split="test",
            ),
            MetricValue(
                "info_dimension_wise_kl_nats",
                dwkl,
                n_samples=n,
                split="test",
            ),
            MetricValue(
                "info_decomposition_residual_nats",
                decomposition_residual,
                n_samples=n,
                split="test",
            ),
            MetricValue(
                "info_decomposition_rate_reference_nats",
                rate_reference,
                n_samples=n,
                split="test",
            ),
        ],
        artifacts={
            "info_mutual_information_estimator": estimator,
            "info_mutual_information_sample_source": source,
            "info_total_correlation_beta_tcvae_mixture_estimator": estimator,
            "info_total_correlation_beta_tcvae_mixture_sample_source": source,
            "info_dimension_wise_kl_estimator": estimator,
            "info_dimension_wise_kl_sample_source": source,
            "info_decomposition_rate_reference": "analytical_conditional_kl",
        },
    )


def compute_autoencodix_total_correlation(varix: Any, z: np.ndarray) -> MetricResult:
    """Compute exact AUTOENCODIX Gaussian covariance total correlation.

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
        Gaussian covariance dependence among latent coordinates; zero means independence.
    Limitations:
        Captures covariance dependence only and is unstable for singular covariance.
    Direction:
        Lower is better for coordinate independence.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        tc = float(varix.visualizer._total_correlation(pd.DataFrame(z)))
    if not np.isfinite(tc):
        warning = (
            "AUTOENCODIX total correlation is undefined because the latent "
            "covariance matrix is singular or contains zero-variance dimensions."
        )
        return MetricResult(
            [
                MetricValue(
                    "info_total_correlation_autoencodix",
                    float("nan"),
                    status="degenerate_input",
                    warning=warning,
                    n_samples=len(z),
                    split="test",
                )
            ],
            warnings=[warning],
        )
    return MetricResult(
        [
            MetricValue(
                "info_total_correlation_autoencodix",
                tc,
                n_samples=len(z),
                split="test",
            )
        ]
    )


def run_information_theoretic_metrics_pipeline(
    inputs: CoreMetricInputs,
    config: InformationMetricConfig,
    varix: Any,
) -> MetricGroupResult:
    if inputs.z_test is None:
        raise ValueError(
            "z_test is required for exact AUTOENCODIX total correlation "
            "(GeneralVisualizer uses result.latentspaces)"
        )
    tc = compute_autoencodix_total_correlation(varix, inputs.z_test)
    if inputs.logvar_test is None:
        warning = "logvar_test is unavailable; MI and dimension-wise KL were skipped."
        return MetricGroupResult(
            "information",
            "skipped_missing_posterior_parameters",
            tc.values,
            artifacts=tc.artifacts,
            warnings=tc.warnings + [warning],
        )
    mixture_components = estimate_beta_tcvae_mixture_information_components(
        inputs.mu_test, inputs.logvar_test, inputs.z_test, config
    )
    results = (mixture_components, tc)
    values = [value for result in results for value in result.values]
    return MetricGroupResult(
        "information",
        "ok" if all(value.status == "ok" for value in values) else "degenerate_input",
        values,
        artifacts={
            key: val for result in results for key, val in result.artifacts.items()
        },
        warnings=[warning for result in results for warning in result.warnings],
    )
