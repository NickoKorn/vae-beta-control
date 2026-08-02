"""Probabilistic posterior diagnostics."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    CoreMetricInputs,
    MetricGroupResult,
    MetricResult,
    MetricValue,
    PosteriorCollapseConfig,
    ProbabilisticMetricConfig,
)

ACTIVE_UNIT_EPSILONS = (0.001, 0.01, 0.1)


def _epsilon_token(epsilon: float) -> str:
    return f"{epsilon:g}".replace(".", "p")


def compute_active_units(mu: np.ndarray, run_name: str) -> MetricResult:
    """Compute active posterior-mean units at three reporting thresholds.

    Metric family:
        Probabilistic posterior diagnostics.
    Space comparison:
        Posterior mean only.
    Exact spaces:
        Empirical posterior-mean representation mu_test over test observations.
    Representations:
        Per-dimension posterior means and their sample variance across observations.
    Mathematical definition:
        A_j=Var_x[mu_j(x)]; unit j is active at epsilon when A_j>epsilon.
    Scientific source:
        Burda, Grosse, and Salakhutdinov, Importance Weighted Autoencoders,
        ICLR 2016, Appendix C.
    Repository provenance:
        NumPy implementation of the cited activity statistic with the paper's
        0.01 threshold plus 0.001 and 0.1 sensitivity thresholds.
    Interpretation:
        Counts coordinates whose posterior means vary materially across samples.
    Limitations:
        Thresholds are scale-dependent reporting conventions; activity does
        not prove semantic usefulness or independence.
    Direction:
        Descriptive; fewer active units can indicate selective posterior collapse.
    """
    values = np.asarray(mu, dtype=np.float64)
    variances = np.var(values, axis=0)
    metrics: list[MetricValue] = []
    for epsilon in ACTIVE_UNIT_EPSILONS:
        active = variances > epsilon
        token = _epsilon_token(epsilon)
        metrics.extend(
            [
                MetricValue(
                    f"posterior_active_units_count_eps_{token}",
                    int(np.sum(active)),
                    n_samples=len(values),
                    split="test",
                ),
                MetricValue(
                    f"posterior_active_units_fraction_eps_{token}",
                    float(np.mean(active)),
                    n_samples=len(values),
                    split="test",
                ),
            ]
        )
    table = pd.DataFrame(
        {
            "run_name": run_name,
            "latent_dimension": np.arange(values.shape[1], dtype=int),
            "posterior_active_units_variance": variances,
        }
    )
    for epsilon in ACTIVE_UNIT_EPSILONS:
        table[f"is_active_eps_{_epsilon_token(epsilon)}"] = variances > epsilon
    return MetricResult(metrics, tables={"active_units_per_dimension": table})


def compute_conditional_kl_per_sample_per_dimension(
    mu: np.ndarray, logvar: np.ndarray, run_name: str
) -> MetricResult:
    """Compute analytical conditional KL for every observation and latent dimension.

    Metric family:
        Probabilistic posterior diagnostics.
    Space comparison:
        Posterior versus prior.
    Exact spaces:
        Diagonal-Gaussian q_phi(z_i|x_n) and standard-normal p(z_i).
    Representations:
        Posterior means mu, posterior log variances, and the standard-normal prior.
    Mathematical definition:
        KL_ni=0.5*(mu_ni^2+exp(logvar_ni)-logvar_ni-1).
    Scientific source:
        Kingma and Welling, arXiv:1312.6114; Lucas et al., arXiv:1911.02469.
    Repository provenance:
        Custom analytical implementation of the cited formula.
    Interpretation:
        Conditional posterior divergence from the prior before averaging.
    Limitations:
        This is not aggregated-posterior marginal KL.
    Direction:
        Descriptive; near-zero dimensions may be collapsed.
    """
    kl = 0.5 * (np.square(mu) + np.exp(logvar) - logvar - 1.0)
    dim_means = np.mean(kl, axis=0)
    table = pd.DataFrame(
        {
            "run_name": run_name,
            "latent_dimension": np.arange(mu.shape[1], dtype=int),
            "conditional_kl_mean": dim_means,
            "conditional_kl_median": np.median(kl, axis=0),
            "conditional_kl_p95": np.quantile(kl, 0.95, axis=0),
        }
    )
    return MetricResult(
        values=[
            MetricValue(
                "posterior_conditional_kl_total_mean",
                float(np.mean(np.sum(kl, axis=1))),
                n_samples=len(mu),
                split="test",
            ),
            MetricValue(
                "posterior_conditional_kl_dim_mean",
                float(np.mean(dim_means)),
                n_samples=len(mu),
                split="test",
            ),
            MetricValue(
                "posterior_conditional_kl_dim_median",
                float(np.median(dim_means)),
                n_samples=len(mu),
                split="test",
            ),
            MetricValue(
                "posterior_conditional_kl_dim_min",
                float(np.min(dim_means)),
                n_samples=len(mu),
                split="test",
            ),
            MetricValue(
                "posterior_conditional_kl_dim_max",
                float(np.max(dim_means)),
                n_samples=len(mu),
                split="test",
            ),
        ],
        vectors={"conditional_kl_matrix": kl},
        tables={"conditional_kl_per_dimension": table},
    )


def compute_epsilon_delta_posterior_collapse(
    conditional_kl: np.ndarray,
    run_name: str,
    config: PosteriorCollapseConfig,
) -> MetricResult:
    """Compute the finite-sample epsilon-delta posterior-collapse definition.

    Metric family:
        Probabilistic posterior diagnostics.
    Space comparison:
        Posterior versus prior.
    Exact spaces:
        Conditional q_phi(z_i|x) versus p(z_i), evaluated over empirical observations.
    Representations:
        Per-observation per-dimension conditional KL values.
    Mathematical definition:
        p_hat_i=mean_n[KL_ni<epsilon]; collapsed_i=1[p_hat_i>=1-delta].
    Scientific source:
        Lucas et al., Don't Blame the ELBO!, Section 6.2, arXiv:1911.02469.
    Repository provenance:
        Custom implementation of the cited strict-inequality definition.
    Interpretation:
        Fraction of latent dimensions whose conditional posterior is near the prior for almost all samples.
    Limitations:
        Epsilon is a reporting choice; zero collapse does not prove usefulness.
    Direction:
        Lower collapse is generally preferable for retaining information.
    """
    curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    primary_count = 0
    for epsilon in config.epsilon_values:
        fractions = np.mean(conditional_kl < epsilon, axis=0)
        collapsed = fractions >= 1.0 - config.delta
        for dimension, (fraction, is_collapsed) in enumerate(zip(fractions, collapsed)):
            curve_rows.append(
                {
                    "run_name": run_name,
                    "epsilon": float(epsilon),
                    "delta": float(config.delta),
                    "latent_dimension": dimension,
                    "sample_fraction_below_epsilon": float(fraction),
                    "is_epsilon_delta_collapsed": bool(is_collapsed),
                }
            )
        count = int(np.sum(collapsed))
        summary_rows.append(
            {
                "run_name": run_name,
                "epsilon": float(epsilon),
                "delta": float(config.delta),
                "collapsed_dimensions_count": count,
                "collapsed_dimensions_fraction": float(count / conditional_kl.shape[1]),
            }
        )
        if epsilon == config.primary_epsilon:
            primary_count = count
    return MetricResult(
        values=[
            MetricValue(
                "posterior_collapse_primary_epsilon", float(config.primary_epsilon)
            ),
            MetricValue("posterior_collapse_delta", float(config.delta)),
            MetricValue("posterior_collapsed_dimensions_count", primary_count),
            MetricValue(
                "posterior_collapsed_dimensions_fraction",
                float(primary_count / conditional_kl.shape[1]),
            ),
        ],
        tables={
            "collapse_curve": pd.DataFrame(curve_rows),
            "collapse_summary_curve": pd.DataFrame(summary_rows),
        },
    )


def compute_autoencodix_latent_coverage(varix: Any, z: np.ndarray) -> MetricResult:
    """Compute the original AUTOENCODIX latent coverage.

    Metric family:
        Probabilistic posterior diagnostics.
    Space comparison:
        Latent only.
    Exact spaces:
        Empirical sampled AUTOENCODIX test latent space z_test.
    Representations:
        Sampled latent coordinates z from result.latentspaces.
    Mathematical definition:
        B=floor(n^(1/d)); coverage is the number of occupied joint B^d cells
        divided by B^d.
    Scientific source:
        Joas et al., AUTOENCODIX, DOI:10.1038/s43588-025-00916-4.
    Repository provenance:
        Direct call to ``varix.visualizer._coverage_calc`` on the visualizer
        instance owned by the trained VARIX pipeline.
    Interpretation:
        Quantifies joint occupancy of the observed latent space.
    Limitations:
        Depends on n, d, binning, and outliers; it is not prior fit or semantic quality.
    Direction:
        Higher occupancy is better descriptively; one is not a universal Gaussian target.
    """
    n_samples = len(z)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Coverage calculation fails.*",
            category=UserWarning,
        )
        coverage = float(varix.visualizer._coverage_calc(pd.DataFrame(z)))
    if np.isnan(coverage):
        warning = (
            "AUTOENCODIX coverage is undefined because its bin-count convention "
            "produced fewer than two bins."
        )
        return MetricResult(
            [
                MetricValue(
                    "latent_coverage_autoencodix",
                    coverage,
                    "degenerate_input",
                    warning,
                    n_samples,
                    "test",
                )
            ],
            warnings=[warning],
        )
    return MetricResult(
        [
            MetricValue(
                "latent_coverage_autoencodix",
                coverage,
                n_samples=n_samples,
                split="test",
            )
        ]
    )


def compute_conditional_posterior_entropy(
    logvar: np.ndarray, run_name: str
) -> MetricResult:
    """Compute conditional diagonal-Gaussian posterior entropy.

    Metric family:
        Probabilistic posterior diagnostics.
    Space comparison:
        Posterior only.
    Exact spaces:
        Conditional approximate posterior q_phi(z|x).
    Representations:
        Posterior log variance, variance, and standard deviation.
    Mathematical definition:
        H[q(z|x)]=0.5*sum_i(1+log(2*pi)+logvar_i(x)).
    Scientific source:
        Cover and Thomas, Elements of Information Theory.
    Repository provenance:
        Custom analytical diagonal-Gaussian implementation.
    Interpretation:
        Average conditional posterior spread.
    Limitations:
        Not calibration, predictive uncertainty, epistemic uncertainty, or quality by itself.
    Direction:
        Descriptive only.
    """
    per_dimension_entropy = 0.5 * (1.0 + np.log(2.0 * np.pi) + logvar)
    totals = np.sum(per_dimension_entropy, axis=1)
    variance = np.exp(logvar)
    table = pd.DataFrame(
        {
            "run_name": run_name,
            "latent_dimension": np.arange(logvar.shape[1], dtype=int),
            "posterior_entropy_mean": np.mean(per_dimension_entropy, axis=0),
            "posterior_variance_mean": np.mean(variance, axis=0),
            "posterior_standard_deviation_mean": np.mean(np.sqrt(variance), axis=0),
        }
    )
    return MetricResult(
        values=[
            MetricValue(
                "posterior_entropy_total_mean",
                float(np.mean(totals)),
                n_samples=len(logvar),
                split="test",
            ),
            MetricValue(
                "posterior_entropy_total_median",
                float(np.median(totals)),
                n_samples=len(logvar),
                split="test",
            ),
            MetricValue(
                "posterior_entropy_total_p95",
                float(np.quantile(totals, 0.95)),
                n_samples=len(logvar),
                split="test",
            ),
            MetricValue(
                "posterior_entropy_per_dimension_mean",
                float(np.mean(per_dimension_entropy)),
                n_samples=len(logvar),
                split="test",
            ),
        ],
        tables={"posterior_entropy_per_dimension": table},
    )


def run_probabilistic_metrics_pipeline(
    inputs: CoreMetricInputs,
    config: ProbabilisticMetricConfig,
    varix: Any,
) -> MetricGroupResult:
    if inputs.z_test is None:
        raise ValueError(
            "z_test is required for exact AUTOENCODIX latent coverage "
            "(GeneralVisualizer uses result.latentspaces)"
        )
    coverage = compute_autoencodix_latent_coverage(varix, inputs.z_test)
    active_units = compute_active_units(inputs.mu_test, inputs.run_name)
    if inputs.logvar_test is None:
        warning = "logvar_test is unavailable; posterior KL, collapse, and entropy were skipped."
        return MetricGroupResult(
            "probabilistic",
            "skipped_missing_posterior_parameters",
            coverage.values + active_units.values,
            tables={**coverage.tables, **active_units.tables},
            artifacts={**coverage.artifacts, **active_units.artifacts},
            warnings=coverage.warnings + [warning],
        )
    kl = compute_conditional_kl_per_sample_per_dimension(
        inputs.mu_test, inputs.logvar_test, inputs.run_name
    )
    collapse = compute_epsilon_delta_posterior_collapse(
        kl.vectors["conditional_kl_matrix"], inputs.run_name, config.collapse
    )
    entropy = compute_conditional_posterior_entropy(inputs.logvar_test, inputs.run_name)
    results = (kl, collapse, coverage, entropy, active_units)
    values = [value for result in results for value in result.values]
    return MetricGroupResult(
        "probabilistic",
        "ok" if all(value.status == "ok" for value in values) else "degenerate_input",
        values,
        tables={key: val for result in results for key, val in result.tables.items()},
        artifacts={
            key: val for result in results for key, val in result.artifacts.items()
        },
        warnings=[warning for result in results for warning in result.warnings],
    )
