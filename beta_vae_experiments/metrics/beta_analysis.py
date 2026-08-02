"""Preregistration-compatible cross-run fixed-beta analysis.

The module deliberately keeps utility endpoints separate and refuses to select
beta from test metrics. With one training seed, point differences are reported,
while confidence intervals and claims of reproducibility remain unavailable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

METRIC_ROLES: dict[str, str] = {
    "recon_test_mse": "primary",
    "recon_val_mse": "primary",
    "geometry_mu_trustworthiness_zadu_k10": "primary",
    "geometry_mu_continuity_zadu_k10": "primary",
    "posterior_conditional_kl_total_mean": "secondary",
    "info_mutual_information_nats": "secondary",
    "info_total_correlation_beta_tcvae_mixture_nats": "secondary",
    "info_dimension_wise_kl_nats": "secondary",
    "posterior_collapsed_dimensions_fraction": "secondary",
    "geometry_effective_rank_mu": "secondary",
    "info_decomposition_residual_nats": "sanity_check",
    "info_decomposition_rate_reference_nats": "sanity_check",
}


@dataclass(frozen=True)
class MarginProfile:
    reconstruction: float
    trustworthiness: float
    continuity: float

    def __post_init__(self) -> None:
        if min(self.reconstruction, self.trustworthiness, self.continuity) <= 0:
            raise ValueError("all non-degradation margins must be positive")


@dataclass(frozen=True)
class BetaAnalysisConfig:
    beta_ref: float = 1.0
    alpha: float = 0.05
    primary_k: int = 10
    strong_collapse_fraction: float = 0.8
    margin_profiles: Mapping[str, MarginProfile] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.margin_profiles and set(self.margin_profiles) != {
            "strict",
            "main",
            "liberal",
        }:
            raise ValueError("margin profiles must be strict, main, and liberal")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        if not 0 <= self.strong_collapse_fraction <= 1:
            raise ValueError("strong_collapse_fraction must lie in [0, 1]")


def audit_varix_loss_scaling(
    beta_nominal: float,
    loss_reduction: str,
    default_vae_loss: str,
    input_dimension: int,
    latent_dimension: int,
) -> dict[str, object]:
    """Describe the exact VARIX reduction relative to per-sample-sum D and R."""
    if loss_reduction not in {"sum", "mean"}:
        raise ValueError("loss_reduction must be 'sum' or 'mean'")
    if input_dimension < 1 or latent_dimension < 1:
        raise ValueError("input and latent dimensions must be positive")
    if default_vae_loss == "kl":
        if loss_reduction == "sum":
            a_d, a_r = 1.0, 1.0
        else:
            a_d, a_r = 1.0 / input_dimension, 1.0 / latent_dimension
        beta_effective = beta_nominal * a_r / a_d
        if not np.isclose(beta_nominal, 1.0):
            reference_type = "non_reference_beta"
        elif np.isclose(beta_effective, 1.0):
            reference_type = "canonical_elbo_scaling"
        else:
            reference_type = "nominal_implementation_reference"
    elif default_vae_loss == "mmd":
        a_d = 1.0 if loss_reduction == "sum" else 1.0 / input_dimension
        a_r = np.nan
        beta_effective = np.nan
        reference_type = "incompatible_mmd_objective_family"
    else:
        raise ValueError("default_vae_loss must be 'kl' or 'mmd'")
    return {
        "beta_nominal": float(beta_nominal),
        "beta_effective": float(beta_effective),
        "distortion_scale_a_D": float(a_d),
        "regularizer_scale_a_R": float(a_r),
        "reference_type": reference_type,
        "loss_equation": "recon_loss + beta * variational_loss",
    }


def add_loss_audit_columns(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "beta",
        "loss_reduction",
        "default_vae_loss",
        "input_dimension",
        "latent_dimension",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("loss audit is missing columns: " + ", ".join(sorted(missing)))
    rows = [
        audit_varix_loss_scaling(
            row.beta,
            row.loss_reduction,
            row.default_vae_loss,
            int(row.input_dimension),
            int(row.latent_dimension),
        )
        for row in frame.itertuples(index=False)
    ]
    audited = frame.copy()
    for column in rows[0] if rows else ():
        audited[column] = [row[column] for row in rows]
    audited["experiment_family"] = (
        audited["default_vae_loss"].astype(str)
        + "_"
        + audited["loss_reduction"].astype(str)
    )
    return audited


def information_decomposition_table(frame: pd.DataFrame) -> pd.DataFrame:
    rate_column = (
        "info_decomposition_rate_reference_nats"
        if "info_decomposition_rate_reference_nats" in frame
        else "posterior_conditional_kl_total_mean"
    )
    columns = {
        rate_column: "rate_reference",
        "info_mutual_information_nats": "mutual_information",
        "info_total_correlation_beta_tcvae_mixture_nats": "total_correlation",
        "info_dimension_wise_kl_nats": "dimensionwise_aggregated_kl",
    }
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(
            "information decomposition is missing: " + ", ".join(sorted(missing))
        )
    identifiers = [
        c
        for c in (
            "experiment_family",
            "run_name",
            "beta",
            "beta_effective",
            "training_seed",
        )
        if c in frame
    ]
    result = frame[identifiers + list(columns)].rename(columns=columns).copy()
    result["decomposition_residual"] = result["rate_reference"] - (
        result["mutual_information"]
        + result["total_correlation"]
        + result["dimensionwise_aggregated_kl"]
    )
    result["metric_role"] = "sanity_check"
    result["reference"] = "Chen et al. (2018), NeurIPS 31; residual is project-derived"
    result["estimator_population_status"] = (
        "matched_selected_subset"
        if rate_column == "info_decomposition_rate_reference_nats"
        else "legacy_full_rate_may_not_match_information_subset"
    )
    return result.sort_values(
        [
            c
            for c in ("experiment_family", "beta_effective", "beta", "training_seed")
            if c in result
        ]
    ).reset_index(drop=True)


def mathematical_beta_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Check adjacent ideal-global-optimum implications within compatible families."""
    required = {
        "experiment_family",
        "beta",
        "beta_effective",
        "recon_test_mse",
        "input_dimension",
        "posterior_conditional_kl_total_mean",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "mathematical diagnostics are missing: " + ", ".join(sorted(missing))
        )
    rows: list[dict[str, object]] = []
    grouping = [
        c for c in ("experiment_family", "learning_rate", "training_seed") if c in frame
    ]
    for keys, group in frame.loc[frame["default_vae_loss"].eq("kl")].groupby(
        grouping, dropna=False
    ):
        ordered = group.sort_values("beta_effective")
        for (_, low), (_, high) in zip(
            ordered.iloc[:-1].iterrows(), ordered.iloc[1:].iterrows()
        ):
            d_low = float(low.recon_test_mse) * int(low.input_dimension)
            d_high = float(high.recon_test_mse) * int(high.input_dimension)
            r_low = float(low.posterior_conditional_kl_total_mean)
            r_high = float(high.posterior_conditional_kl_total_mean)
            delta_d, delta_r = d_high - d_low, r_low - r_high
            secant = delta_d / delta_r if delta_r > 0 else np.nan
            beta_low, beta_high = float(low.beta_effective), float(high.beta_effective)
            rows.append(
                {
                    "experiment_family": low.experiment_family,
                    "beta_low": beta_low,
                    "beta_high": beta_high,
                    "D_low": d_low,
                    "D_high": d_high,
                    "R_low": r_low,
                    "R_high": r_high,
                    "delta_D": delta_d,
                    "delta_R": delta_r,
                    "secant_beta": secant,
                    "rate_nonincreasing": r_high <= r_low,
                    "distortion_nondecreasing": d_high >= d_low,
                    "inside_theoretical_interval": bool(
                        np.isfinite(secant) and beta_low <= secant <= beta_high
                    ),
                    "diagnostic_scope": "ideal_global_optimum_consistency_only",
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["experiment_family", "beta_low"], kind="stable")
        .reset_index(drop=True)
        if rows
        else pd.DataFrame()
    )


def collapse_transition(
    frame: pd.DataFrame, config: BetaAnalysisConfig
) -> pd.DataFrame:
    required = {"experiment_family", "beta", "posterior_collapsed_dimensions_fraction"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "collapse transition is missing: " + ", ".join(sorted(missing))
        )
    rows = []
    group_columns = [
        column for column in ("experiment_family", "learning_rate") if column in frame
    ]
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        stratum = dict(zip(group_columns, key_values))
        ordered = group.sort_values("beta")
        strong = ordered["posterior_collapsed_dimensions_fraction"].ge(
            config.strong_collapse_fraction
        )
        first_pos = int(np.flatnonzero(strong)[0]) if strong.any() else None
        first_strong = (
            float(ordered.iloc[first_pos].beta) if first_pos is not None else np.nan
        )
        last_clear = (
            float(ordered.iloc[first_pos - 1].beta)
            if first_pos not in (None, 0)
            else np.nan
        )
        n_seeds = int(ordered.get("training_seed", pd.Series([0])).nunique())
        rows.append(
            {
                **stratum,
                "last_clearly_noncollapsed_beta": last_clear,
                "beta_collapse_low": last_clear,
                "beta_collapse_high": first_strong,
                "first_strongly_collapsed_beta": first_strong,
                "strong_collapse_fraction_threshold": config.strong_collapse_fraction,
                "n_seeds": n_seeds,
                "evidence_status": "single_seed_provisional"
                if n_seeds == 1
                else "multi_seed_reproducibility_assessable",
            }
        )
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def save_single_seed_beta_analysis(
    overview: pd.DataFrame,
    output_dir: Path,
    config: BetaAnalysisConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Persist sorted descriptive tables; never select beta from test endpoints."""
    config = config or BetaAnalysisConfig()
    audited = add_loss_audit_columns(overview)
    decomposition = information_decomposition_table(audited)
    if "info_decomposition_residual_nats" not in audited:
        audited["info_decomposition_residual_nats"] = decomposition[
            "decomposition_residual"
        ].to_numpy()
    mathematical = mathematical_beta_diagnostics(audited)
    collapse = collapse_transition(audited, config)
    id_columns = [
        c
        for c in (
            "experiment_family",
            "run_name",
            "beta",
            "beta_effective",
            "training_seed",
        )
        if c in audited
    ]
    metric_columns = [column for column in METRIC_ROLES if column in audited]
    metric_results = audited.melt(
        id_vars=id_columns,
        value_vars=metric_columns,
        var_name="metric_name",
        value_name="value",
    )
    metric_results["posterior_sample_id"] = pd.NA
    metric_results["split"] = np.where(
        metric_results["metric_name"].str.contains("_val_"), "validation", "test"
    )
    metric_results["metric_family"] = metric_results["metric_name"].map(
        lambda name: name.split("_", 1)[0]
    )
    metric_results["metric_role"] = metric_results["metric_name"].map(METRIC_ROLES)
    metric_results["status"] = np.where(
        np.isfinite(pd.to_numeric(metric_results["value"], errors="coerce")),
        "ok",
        "non_finite",
    )
    metric_results = metric_results.sort_values(
        [
            "experiment_family",
            "beta_effective",
            "beta",
            "training_seed",
            "metric_role",
            "metric_name",
        ],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    summary_rows: list[dict[str, object]] = []
    for (family, metric_name), group in metric_results.groupby(
        ["experiment_family", "metric_name"], sort=True
    ):
        reference = group.loc[np.isclose(group["beta"].astype(float), config.beta_ref)]
        reference_value = (
            float(reference.iloc[0].value) if len(reference) == 1 else np.nan
        )
        higher_is_better = metric_name in {
            "geometry_mu_trustworthiness_zadu_k10",
            "geometry_mu_continuity_zadu_k10",
            "geometry_effective_rank_mu",
        }
        for row in group.itertuples(index=False):
            value = float(row.value)
            degradation = (
                (
                    reference_value - value
                    if higher_is_better
                    else value - reference_value
                )
                if np.isfinite(reference_value)
                else np.nan
            )
            summary_rows.append(
                {
                    "experiment_family": family,
                    "beta": row.beta,
                    "beta_effective": row.beta_effective,
                    "metric_name": metric_name,
                    "n_seeds": 1,
                    "mean": value,
                    "median": value,
                    "std": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "mean_difference_vs_ref": degradation,
                    "inference_status": "single_seed_ci_not_estimable",
                }
            )
    beta_summary = (
        pd.DataFrame(summary_rows)
        .sort_values(
            ["experiment_family", "beta_effective", "beta", "metric_name"],
            kind="stable",
            na_position="last",
        )
        .reset_index(drop=True)
    )

    decision_rows = []
    main = config.margin_profiles.get("main")
    for beta in sorted(audited["beta"].dropna().unique()):
        for endpoint, margin in (
            ("reconstruction", main.reconstruction if main else np.nan),
            ("trustworthiness", main.trustworthiness if main else np.nan),
            ("continuity", main.continuity if main else np.nan),
        ):
            decision_rows.append(
                {
                    "beta": float(beta),
                    "endpoint": endpoint,
                    "margin": margin,
                    "mean_degradation": np.nan,
                    "degradation_ucb": np.nan,
                    "passes": pd.NA,
                    "status": (
                        "not_assessable_single_seed_or_missing_validation_metric"
                        if main
                        else "not_assessable_margins_not_preregistered"
                    ),
                    "selection_split": "validation",
                }
            )
    decisions = pd.DataFrame(decision_rows)
    selection = pd.DataFrame(
        {
            "beta_ref": [config.beta_ref],
            "reference_type": ["implementation_reference_pending_validation"],
            "safe_tested_betas": [""],
            "beta_max_non_degraded": [np.nan],
            "beta_knee": [np.nan],
            "beta_collapse_low": [
                collapse["beta_collapse_low"].min() if not collapse.empty else np.nan
            ],
            "beta_collapse_high": [
                collapse["beta_collapse_high"].min() if not collapse.empty else np.nan
            ],
            "limiting_primary_endpoint": ["not_assessable"],
            "selection_status": ["locked_single_seed_and_missing_validation_geometry"],
        }
    )
    tables = {
        "metric_results": metric_results,
        "beta_summary": beta_summary,
        "primary_endpoint_decision": decisions,
        "beta_selection_summary": selection,
        "mathematical_beta_diagnostics": mathematical,
        "collapse_transition": collapse,
        "information_decomposition": decomposition,
    }
    destination = Path(output_dir) / "fixed_beta_analysis"
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(destination / f"{name}.csv", index=False)
    return tables


__all__ = [
    "METRIC_ROLES",
    "BetaAnalysisConfig",
    "MarginProfile",
    "add_loss_audit_columns",
    "audit_varix_loss_scaling",
    "collapse_transition",
    "information_decomposition_table",
    "mathematical_beta_diagnostics",
    "save_single_seed_beta_analysis",
]
