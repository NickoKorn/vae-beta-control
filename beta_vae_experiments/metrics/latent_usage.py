"""Diagnostics describing how a trained VARIX model uses its latent space.

Strict literature-aligned core:
- Active Units use the Burda et al. threshold Var_x[mu_j(x)] > 1e-2.
- Posterior collapse uses the Lucas et al. finite-sample (epsilon, delta) criterion,
  with epsilon=1e-2 and delta=1e-2 as the primary reported setting.

Sensitivity-only diagnostics:
- Additional Active-Unit thresholds are persisted only in the per-dimension artifact.
- Additional epsilon values are persisted only in the collapse-curve artifacts.
- AUTOENCODIX latent coverage is reported explicitly as a sensitivity/confirmatory
  diagnostic using the pinned AUTOENCODIX implementation.
- Pairwise Gaussian Hellinger distances describe posterior distinguishability,
  not a thresholded collapse criterion or an annealing control rule.

No project-defined analytical beta phase-transition boundary is reported here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd

from .information_theoretic import compute_conditional_kl_matrix

if TYPE_CHECKING:
    from ..metric_pipeline import CoreMetricInputs


PRIMARY_ACTIVE_UNIT_EPSILON = 1e-2
ACTIVE_UNIT_SENSITIVITY_EPSILONS = (1e-3, 1e-1)


@dataclass(frozen=True)
class PosteriorCollapseConfig:
    """Configuration for the Lucas et al. epsilon-delta collapse criterion.

    A latent coordinate j is classified as collapsed when

        P_x[ KL(q(z_j|x) || p(z_j)) < epsilon ] >= 1 - delta.

    The primary setting is epsilon=0.01 and delta=0.01. Additional epsilon
    values are retained only for sensitivity curves and are not promoted to
    independent core metrics.
    """

    delta: float = 0.01
    epsilon_values: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    primary_epsilon: float = 1e-2

    def __post_init__(self) -> None:
        object.__setattr__(self, "epsilon_values", tuple(self.epsilon_values))
        if not 0 <= self.delta < 1:
            raise ValueError("collapse delta must satisfy 0 <= delta < 1")
        if not self.epsilon_values or any(value <= 0 for value in self.epsilon_values):
            raise ValueError("collapse epsilon values must be positive")
        if self.primary_epsilon not in self.epsilon_values:
            raise ValueError("primary_epsilon must occur in epsilon_values")


@dataclass(frozen=True)
class LatentUsageMetricConfig:
    """Configuration namespace for literature-aligned latent-usage diagnostics."""

    collapse: PosteriorCollapseConfig = field(default_factory=PosteriorCollapseConfig)
    hellinger_max_pairs: int = 10000
    hellinger_seed: int = 31

    def __post_init__(self) -> None:
        if type(self.hellinger_max_pairs) is not int or self.hellinger_max_pairs < 1:
            raise ValueError("hellinger_max_pairs must be a positive integer")
        if type(self.hellinger_seed) is not int or not 0 <= self.hellinger_seed < 2**32:
            raise ValueError("hellinger_seed must be an integer in [0, 2**32)")


SQFA_COMMIT = "5d66578e9e4d497c58c6dbdd5af3b37789cfe04a"
SQFA_VERSION = "0.2.1.dev23+g5d66578e9"
HELLINGER_CONVENTION = "SQFA raw sqrt(1-exp(-D_B)+1e-6); identical posteriors: 0.001"


def diagonal_gaussian_hellinger(mu_a, logvar_a, mu_b, logvar_b):
    """Measure posterior overlap using SQFA's Gaussian Hellinger distance.

    Applicable because our VAE posteriors are Gaussians with covariance
    diag(exp(logvar)). Smaller distances mean greater overlap, not necessarily
    posterior collapse. SQFA's epsilon makes identical posteriors score 0.001.
    Source: https://github.com/dherrera1911/sqfa/tree/5d66578e9e4d497c58c6dbdd5af3b37789cfe04a
    """
    import torch
    from importlib.metadata import version
    from sqfa import distances as sqfa_distances

    if version('sqfa') != SQFA_VERSION:
        raise ValueError('SQFA version differs from the audited requirements pin')
    if sqfa_distances.EPSILON != 1e-6:
        raise ValueError('SQFA Hellinger epsilon differs from the audited convention')
    a, va, b, vb = [np.asarray(v, dtype=np.float64) for v in
                     (mu_a, logvar_a, mu_b, logvar_b)]
    if (a.ndim != 2 or a.shape[1] == 0
            or any(v.shape != a.shape for v in (va, b, vb))):
        raise ValueError("Posterior parameters must have matching (pairs, dimensions) shapes")
    if not all(np.isfinite(v).all() for v in (a, va, b, vb)):
        raise ValueError("Posterior parameters must be finite")
    with np.errstate(over='ignore', under='ignore'):
        variance_a, variance_b = np.exp(va), np.exp(vb)
    if any(not np.isfinite(v).all() or (v <= 0).any() for v in (variance_a, variance_b)):
        raise ValueError('Posterior variances are not representable as finite positive float64 values')
    result = np.empty(len(a), dtype=np.float64)
    # SQFA compares all A/B combinations. Small blocks bound that temporary
    # matrix; only diagonal entries correspond to our selected aligned pairs.
    with torch.no_grad():
        for start in range(0, len(a), 16):
            stop = min(start + 16, len(a))
            stats_a = dict(means=torch.from_numpy(a[start:stop]),
                covariances=torch.diag_embed(torch.from_numpy(variance_a[start:stop])))
            stats_b = dict(means=torch.from_numpy(b[start:stop]),
                covariances=torch.diag_embed(torch.from_numpy(variance_b[start:stop])))
            try:
                matrix = sqfa_distances.hellinger(stats_a, stats_b)
            except RuntimeError as exc:
                raise ValueError(f'SQFA Hellinger failed: {exc}') from exc
            block_size = stop - start
            result[start:stop] = matrix.reshape(block_size, block_size).diagonal().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError('SQFA returned nonfinite Hellinger distances; no clipping or fallback')
    return result


def compute_posterior_hellinger(mu, logvar, config, artifact_dir=None, sample_ids=None):
    """Summarize uniformly selected distinct unordered posterior pairs.

    Fixed seed selects the same row pairs across runs with the same ordered
    cohort. All pairs are used when within budget; no n-by-n matrix is built.
    Quantiles describe the selected pairs, not confidence intervals. Evaluate
    on validation data separately before using this diagnostic for control.
    """
    values = np.asarray(mu, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("mu must be a non-empty matrix")
    n = len(values)
    names = ('mean', 'median', 'q10', 'q90')
    status, warning = 'ok', HELLINGER_CONVENTION
    summary = np.full(4, np.nan)
    if logvar is None:
        status, warning = 'missing_posterior', 'Posterior log-variances are unavailable'
    elif n < 2:
        status, warning = 'degenerate_input', 'Hellinger requires at least two samples'
    else:
        variances = np.asarray(logvar, dtype=np.float64)
        if variances.shape != values.shape:
            raise ValueError("mu and logvar must have matching shapes")
        total = n * (n - 1) // 2
        count = min(total, config.hellinger_max_pairs)
        ranks = (np.arange(total) if count == total else np.sort(
            np.random.default_rng(config.hellinger_seed).choice(total, count, replace=False)))
        starts = np.arange(n, dtype=np.int64)
        starts = starts * (2 * n - starts - 1) // 2
        i = np.searchsorted(starts, ranks, side='right') - 1
        j = i + 1 + ranks - starts[i]
        try:
            distances = diagonal_gaussian_hellinger(values[i], variances[i], values[j], variances[j])
            if not np.isfinite(distances).all():
                raise ValueError('Nonfinite analytic Hellinger distances')
            summary = [np.mean(distances), np.median(distances),
                       *np.quantile(distances, [.1, .9])]
            if artifact_dir is not None:
                import json
                directory = Path(artifact_dir)
                directory.mkdir(parents=True, exist_ok=True)
                pairs = pd.DataFrame(dict(row_i=i, row_j=j, hellinger=distances))
                if sample_ids is not None:
                    ids = np.asarray(sample_ids)
                    if ids.shape != (n,):
                        raise ValueError('sample_ids must align with posterior rows')
                    pairs['sample_id_i'], pairs['sample_id_j'] = ids[i], ids[j]
                pairs.to_csv(directory / 'posterior_hellinger_pairs.csv', index=False)
                (directory / 'posterior_hellinger_metadata.json').write_text(json.dumps(dict(
                    seed=config.hellinger_seed, max_pairs=config.hellinger_max_pairs,
                    n_pairs=count, total_pairs=total, latent_dimensions=values.shape[1],
                    backend='sqfa.distances.hellinger', package_version=SQFA_VERSION,
                    source_commit=SQFA_COMMIT, epsilon=1e-6, dtype='float64',
                    convention=HELLINGER_CONVENTION, batch_size=16,
                    pair_selection='uniform distinct unordered row pairs, without replacement',
                ), indent=2) + '\n')
        except ImportError as exc:
            status, warning = 'missing_dependency', str(exc)
            summary = np.full(4, np.nan)
        except ValueError as exc:
            status, warning = 'degenerate_input', str(exc)
            summary = np.full(4, np.nan)
    return pd.DataFrame([dict(metric_name='posterior_pairwise_hellinger_' + name,
        value=float(value), status=status, warning=warning, n_samples=n, split='test')
        for name, value in zip(names, summary)])


def _epsilon_token(epsilon: float) -> str:
    return f"{epsilon:g}".replace(".", "p")


def compute_epsilon_delta_posterior_collapse(
    conditional_kl: np.ndarray,
    run_name: str,
    config: PosteriorCollapseConfig,
    artifact_dir: Path | None = None,
) -> pd.DataFrame:
    """Apply the finite-sample epsilon-delta posterior-collapse criterion.

    Input representation:
        Sample-by-latent-dimension conditional KL matrix.

    Mathematical definition:
        Coordinate j is collapsed when the empirical fraction of samples with

            KL(q(z_j|x) || p(z_j)) < epsilon

        is at least 1-delta.

    Scientific source:
        Lucas et al., "Understanding Posterior Collapse in Generative Latent
        Variable Models", ICLR 2019 workshop (https://openreview.net/forum?id=r1xaVLUYuE).

    Primary reported setting:
        epsilon=0.01 and delta=0.01.

    Sensitivity:
        Additional configured epsilon values are written to curve artifacts only.
        They are not returned as separate core scalar metrics.

    Interpretation:
        The returned count/fraction quantify coordinates whose conditional
        posterior is near the prior for almost all evaluated samples.

    Limitations:
        Threshold- and split-dependent; does not itself measure downstream utility.

    Direction:
        A larger collapsed fraction indicates stronger explicit posterior collapse.
    """
    matrix = np.asarray(conditional_kl, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("conditional_kl must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("conditional_kl must contain only finite values")

    curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    primary_count: int | None = None

    for epsilon in config.epsilon_values:
        fractions = np.mean(matrix < epsilon, axis=0)
        collapsed = fractions >= 1.0 - config.delta

        for dimension, (fraction, is_collapsed) in enumerate(zip(fractions, collapsed)):
            curve_rows.append(
                {
                    "run_name": run_name,
                    "epsilon": float(epsilon),
                    "delta": float(config.delta),
                    "latent_dimension": int(dimension),
                    "sample_fraction_below_epsilon": float(fraction),
                    "is_epsilon_delta_collapsed": bool(is_collapsed),
                    "is_primary_epsilon": bool(epsilon == config.primary_epsilon),
                }
            )

        count = int(np.sum(collapsed))
        summary_rows.append(
            {
                "run_name": run_name,
                "epsilon": float(epsilon),
                "delta": float(config.delta),
                "collapsed_dimensions_count": count,
                "collapsed_dimensions_fraction": float(count / matrix.shape[1]),
                "is_primary_epsilon": bool(epsilon == config.primary_epsilon),
            }
        )

        if epsilon == config.primary_epsilon:
            primary_count = count

    if primary_count is None:
        raise RuntimeError("primary epsilon was not evaluated")

    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for filename, frame in (
            ("collapse_curve.csv", pd.DataFrame(curve_rows)),
            ("collapse_summary_curve.csv", pd.DataFrame(summary_rows)),
        ):
            path = artifact_dir / filename
            frame.to_csv(path, index=False)
            print(path)

    return pd.DataFrame.from_records(
        [
            {
                "metric_name": f"posterior_collapsed_dimensions_count_eps_{_epsilon_token(config.primary_epsilon)}_delta_{_epsilon_token(config.delta)}",
                "value": primary_count,
                "status": "ok",
                "warning": None,
                "n_samples": len(matrix),
                "split": "test",
            },
            {
                "metric_name": f"posterior_collapsed_dimensions_fraction_eps_{_epsilon_token(config.primary_epsilon)}_delta_{_epsilon_token(config.delta)}",
                "value": float(primary_count / matrix.shape[1]),
                "status": "ok",
                "warning": None,
                "n_samples": len(matrix),
                "split": "test",
            },
        ]
    )


def compute_active_units(
    mu: np.ndarray,
    run_name: str,
    artifact_dir: Path | None = None,
) -> pd.DataFrame:
    """Count Active Units using the Burda et al. convention.

    Input representation:
        Saved posterior means ``mu_test``.

    Mathematical definition:
        For coordinate j,

            A_j = Var_x[E_{q(z_j|x)} z_j] = Var_x[mu_j(x)].

        Burda et al. classify a unit as active when

            A_j > 1e-2.

    Scientific source:
        Burda, Grosse, and Salakhutdinov,
        "Importance Weighted Autoencoders", arXiv:1509.00519.

    Core metric:
        Only the count at epsilon=0.01 is returned as a core scalar.

    Sensitivity:
        Thresholds 0.001 and 0.1 are persisted in the per-dimension artifact
        for sensitivity analysis only and are not promoted to core metrics.

    Interpretation:
        Counts latent coordinates whose posterior mean varies across observations.

    Limitations:
        Threshold-dependent, insensitive to posterior variance, and not itself
        an information or downstream-utility measure.

    Direction:
        Descriptive; more active units are not automatically preferable.
    """
    values = np.asarray(mu, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("mu must be a non-empty two-dimensional matrix")
    if not np.isfinite(values).all():
        raise ValueError("mu must contain only finite values")

    variances = np.var(values, axis=0, ddof=0)
    active_primary = variances > PRIMARY_ACTIVE_UNIT_EPSILON

    table = pd.DataFrame(
        {
            "run_name": run_name,
            "latent_dimension": np.arange(values.shape[1], dtype=int),
            "posterior_active_units_variance": variances,
            f"is_active_eps_{_epsilon_token(PRIMARY_ACTIVE_UNIT_EPSILON)}": active_primary,
        }
    )

    for epsilon in ACTIVE_UNIT_SENSITIVITY_EPSILONS:
        table[f"is_active_eps_{_epsilon_token(epsilon)}_sensitivity"] = variances > epsilon

    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "active_units_per_dimension.csv"
        table.to_csv(path, index=False)
        print(path)

    return pd.DataFrame.from_records(
        [
            {
                "metric_name": "posterior_active_units_count_eps_0p01",
                "value": int(np.sum(active_primary)),
                "status": "ok",
                "warning": None,
                "n_samples": len(values),
                "split": "test",
            }
        ]
    )


def compute_autoencodix_latent_coverage(
    varix: Any,
    z: np.ndarray,
) -> pd.DataFrame:
    """Delegate latent-space coverage to the pinned AUTOENCODIX implementation.

    Input representation:
        Persisted sampled latent coordinates ``z_test``.

    Metric family:
        AUTOENCODIX latent-space coverage sensitivity diagnostic.

    Scientific source:
        AUTOENCODIX (Joas et al.).

    Repository provenance:
        Direct call to ``varix.visualizer._coverage_calc`` from the pinned
        AUTOENCODIX implementation. The project does not reimplement or modify
        the coverage formula.

    Interpretation:
        Reports latent-space coverage according to AUTOENCODIX's published
        binning/coverage convention.

    Limitations:
        Sensitive to sample count, binning, latent dimensionality, outliers,
        and the particular sampled ``z`` realization.

    Direction:
        Descriptive; higher coverage is not automatically a better representation.

    Core status:
        Sensitivity/confirmatory only; not treated as an independent core
        representation-quality metric.
    """
    latent_samples = np.asarray(z, dtype=np.float64)
    if latent_samples.ndim != 2 or latent_samples.shape[0] == 0 or latent_samples.shape[1] == 0:
        raise ValueError("z must be a non-empty two-dimensional matrix")
    if not np.isfinite(latent_samples).all():
        raise ValueError("z must contain only finite values")

    n_samples = len(latent_samples)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Coverage calculation fails.*",
            category=UserWarning,
        )
        coverage = float(
            varix.visualizer._coverage_calc(pd.DataFrame(latent_samples))
        )

    metric_name = "latent_coverage_autoencodix_sensitivity"

    if not np.isfinite(coverage):
        warning = (
            "AUTOENCODIX coverage is undefined under the implementation's "
            "binning convention for this latent sample."
        )
        return pd.DataFrame.from_records(
            [
                {
                    "metric_name": metric_name,
                    "value": float("nan"),
                    "status": "degenerate_input",
                    "warning": warning,
                    "n_samples": n_samples,
                    "split": "test",
                }
            ]
        )

    return pd.DataFrame.from_records(
        [
            {
                "metric_name": metric_name,
                "value": coverage,
                "status": "ok",
                "warning": (
                    "Sensitivity/confirmatory AUTOENCODIX coverage; "
                    "not an independent core quality score."
                ),
                "n_samples": n_samples,
                "split": "test",
            }
        ]
    )


def run_latent_usage_metrics_pipeline(
    inputs: CoreMetricInputs,
    config: LatentUsageMetricConfig,
    varix: Any,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Evaluate literature-aligned latent usage and posterior-collapse diagnostics.

    Core:
        - Active Units count at Var_x(mu_j) > 0.01.
        - Lucas et al. epsilon-delta collapsed-dimension count/fraction at the
          configured primary epsilon/delta (default 0.01/0.01).

    Sensitivity/confirmatory:
        - Pairwise Hellinger mean/median/q10/q90 using posterior means/logvars;
          missing logvars produce explicit unavailable values.
        - AUTOENCODIX latent coverage from sampled ``z_test``.
        - Alternative Active-Unit thresholds in the artifact only.
        - Alternative epsilon values in collapse-curve artifacts only.

    No project-defined analytical beta phase-transition threshold is computed.
    """
    artifact_dir = (
        Path(output_dir) / "absolute_core_metrics_v1" / "posterior" / inputs.run_name
        if output_dir is not None
        else None
    )

    frames: list[pd.DataFrame] = [
        compute_active_units(inputs.mu_test, inputs.run_name, artifact_dir),
        compute_posterior_hellinger(inputs.mu_test, inputs.logvar_test, config,
            artifact_dir, getattr(inputs, 'sample_ids_test', None)),
    ]

    if inputs.z_test is not None:
        frames.append(
            compute_autoencodix_latent_coverage(varix, inputs.z_test)
        )

    if inputs.logvar_test is not None:
        conditional_kl = compute_conditional_kl_matrix(
            inputs.mu_test,
            inputs.logvar_test,
        )
        frames.append(
            compute_epsilon_delta_posterior_collapse(
                conditional_kl,
                inputs.run_name,
                config.collapse,
                artifact_dir,
            )
        )

    return pd.concat(frames, ignore_index=True)


__all__ = [
    "diagonal_gaussian_hellinger",
    "compute_posterior_hellinger",
    "compute_active_units",
    "compute_autoencodix_latent_coverage",
    "compute_epsilon_delta_posterior_collapse",
    "run_latent_usage_metrics_pipeline",
]
