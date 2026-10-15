"""Active sweep analysis for rate-distortion and posterior-collapse boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TRAJECTORY_KEYS = (
    "dataset_id",
    "learning_rate",
    "latent_dim",
    "seed",
    "epochs",
    "loss",
    "loss_reduction",
    "anneal_function",
)
AGGREGATE_KEYS = (
    "dataset_id",
    "learning_rate",
    "latent_dim",
    "epochs",
    "loss",
    "loss_reduction",
    "anneal_function",
)
ACTIVE_TRAJECTORY_METRICS = (
    "info_total_correlation_autoencodix",
    "posterior_snr_mean",
    "dimensionality_mu_effective_rank_fraction",
    "geometry_mu_lcmc_zadu_k10",
    "geometry_mu_lcmc_zadu_k20",
    "topology_nts_m",
    "topology_srtd_lite",
    "topology_nts_e",
    "topology_h0_normalized_wasserstein",
    "topology_h1_normalized_wasserstein",
    "dtm_h0_wasserstein",
    "dtm_h1_wasserstein",
    "posterior_collapsed_dimensions_fraction",
    "posterior_conditional_kl_total_mean",
    "autoencodix_valid_recon_loss",
)


def _available_keys(frame: pd.DataFrame, keys: tuple[str, ...]) -> tuple[str, ...]:
    """Use optional training identifiers when the input table records them."""
    return tuple(key for key in keys if key in frame.columns)


def _finite_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan)


def _mark_beta_states(axis, frame: pd.DataFrame) -> None:
    """Mark beta reference states only when they are inferable from the data."""
    if "beta" not in frame or frame.empty:
        return
    numeric_columns = [
        column for column in (
            "posterior_conditional_kl_total_mean", "autoencodix_valid_recon_loss",
            "downstream_mean_roc_auc_ovo", "posterior_collapsed_dimensions_fraction",
        ) if column in frame
    ]
    data = _finite_numeric(frame, ("beta", *numeric_columns))
    aggregate = data.groupby("beta", as_index=False)[numeric_columns].median().sort_values("beta")
    candidates: list[tuple[float, str, str]] = []
    if np.isclose(aggregate["beta"], 0).any():
        candidates.append((0.0, "beta=0", "black"))
    if "downstream_mean_roc_auc_ovo" in aggregate and aggregate["downstream_mean_roc_auc_ovo"].notna().any():
        index = aggregate["downstream_mean_roc_auc_ovo"].idxmax()
        candidates.append((float(aggregate.loc[index, "beta"]), "evaluation peak", "green"))
    elif "autoencodix_valid_recon_loss" in aggregate and aggregate["autoencodix_valid_recon_loss"].notna().any():
        index = aggregate["autoencodix_valid_recon_loss"].idxmin()
        candidates.append((float(aggregate.loc[index, "beta"]), "best reconstruction", "green"))
    if "posterior_conditional_kl_total_mean" in aggregate:
        drops = -aggregate["posterior_conditional_kl_total_mean"].diff()
        if drops.notna().any() and drops.max() > 0:
            index = drops.idxmax()
            candidates.append((float(aggregate.loc[index, "beta"]), "rate drop", "orange"))
    if "posterior_collapsed_dimensions_fraction" in aggregate:
        collapsed = aggregate.loc[aggregate["posterior_collapsed_dimensions_fraction"] >= 0.1]
        if not collapsed.empty:
            candidates.append((float(collapsed.iloc[0]["beta"]), "collapse onset", "red"))
    seen: set[tuple[float, str]] = set()
    for beta, label, color in candidates:
        if (beta, label) not in seen:
            axis.axvline(beta, color=color, linestyle="--", alpha=0.5, label=label)
            seen.add((beta, label))


def pareto_mask(rate: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    """Return non-dominated finite points when rate and distortion are minimized."""
    rate = np.asarray(rate, dtype=float)
    distortion = np.asarray(distortion, dtype=float)
    keep = np.zeros(len(rate), dtype=bool)
    finite = np.isfinite(rate) & np.isfinite(distortion)
    for index in np.flatnonzero(finite):
        candidates = finite & (rate <= rate[index]) & (distortion <= distortion[index])
        strict = (rate < rate[index]) | (distortion < distortion[index])
        keep[index] = not np.any(candidates & strict)
    return keep


def local_rd_slopes(rate: np.ndarray, distortion: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Compute forward local -delta-D/delta-R with a near-zero-rate guard."""
    rate = np.asarray(rate, dtype=float)
    distortion = np.asarray(distortion, dtype=float)
    slopes = np.full(len(rate), np.nan)
    if len(rate) > 1:
        delta_rate = np.diff(rate)
        delta_distortion = np.diff(distortion)
        valid = np.isfinite(delta_rate) & np.isfinite(delta_distortion) & (np.abs(delta_rate) > epsilon)
        slopes[:-1][valid] = -delta_distortion[valid] / delta_rate[valid]
    return slopes


def build_rate_distortion_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Annotate each fixed-seed beta trajectory with R-D diagnostics."""
    trajectory_keys = _available_keys(frame, TRAJECTORY_KEYS)
    required = {
        "dataset_id", "learning_rate", "latent_dim", "seed", "beta",
        "posterior_conditional_kl_total_mean", "autoencodix_valid_recon_loss",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing rate-distortion columns: {sorted(missing)}")
    data = _finite_numeric(frame, ("beta", "posterior_conditional_kl_total_mean", "autoencodix_valid_recon_loss"))
    rows = []
    for _, group in data.groupby(list(trajectory_keys), dropna=False, sort=True):
        group = group.sort_values("beta").copy()
        if group["beta"].duplicated().any():
            raise ValueError("Duplicate beta values in rate-distortion trajectory")
        rate = group["posterior_conditional_kl_total_mean"].to_numpy(float)
        distortion = group["autoencodix_valid_recon_loss"].to_numpy(float)
        group["rd_is_pareto_optimal"] = pareto_mask(rate, distortion)
        group["rd_local_slope"] = local_rd_slopes(rate, distortion)
        rows.append(group)
    return pd.concat(rows, ignore_index=True) if rows else data.iloc[0:0]


def collapse_boundaries(frame: pd.DataFrame) -> pd.DataFrame:
    """Find first and persistent 0.1/0.5 collapse thresholds without imputation."""
    trajectory_keys = _available_keys(frame, TRAJECTORY_KEYS)
    aggregate_keys = _available_keys(frame, AGGREGATE_KEYS)
    rows = []
    for keys, group in frame.groupby(list(trajectory_keys), dropna=False, sort=True):
        ordered = group.sort_values("beta")
        beta = pd.to_numeric(ordered["beta"], errors="coerce").to_numpy(float)
        collapse = pd.to_numeric(ordered["posterior_collapsed_dimensions_fraction"], errors="coerce").to_numpy(float)
        row = dict(zip(trajectory_keys, keys))
        for threshold, label in ((0.1, "0p1"), (0.5, "0p5")):
            hits = np.flatnonzero(np.isfinite(collapse) & (collapse >= threshold))
            persistent = np.flatnonzero(np.isfinite(collapse[:-1]) & np.isfinite(collapse[1:]) & (collapse[:-1] >= threshold) & (collapse[1:] >= threshold))
            row[f"beta_collapse_{label}"] = float(beta[hits[0]]) if len(hits) else np.nan
            row[f"beta_collapse_{label}_censored"] = not bool(len(hits))
            row[f"beta_collapse_{label}_persistent"] = float(beta[persistent[0]]) if len(persistent) else np.nan
            row[f"beta_collapse_{label}_persistent_censored"] = not bool(len(persistent))
        rows.append(row)
    seed_rows = pd.DataFrame(rows)
    aggregate = frame.groupby([*aggregate_keys, "beta"], dropna=False, as_index=False)["posterior_collapsed_dimensions_fraction"].median()
    aggregate["seed"] = np.nan
    aggregate_rows = []
    for keys, group in aggregate.groupby(list(aggregate_keys), dropna=False, sort=True):
        ordered = group.sort_values("beta")
        beta = ordered["beta"].to_numpy(float)
        collapse = ordered["posterior_collapsed_dimensions_fraction"].to_numpy(float)
        row = {**dict(zip(aggregate_keys, keys)), "seed": np.nan}
        for threshold, label in ((0.1, "0p1"), (0.5, "0p5")):
            hits = np.flatnonzero(np.isfinite(collapse) & (collapse >= threshold))
            persistent = np.flatnonzero((collapse[:-1] >= threshold) & (collapse[1:] >= threshold))
            row[f"beta_collapse_{label}"] = float(beta[hits[0]]) if len(hits) else np.nan
            row[f"beta_collapse_{label}_censored"] = not bool(len(hits))
            row[f"beta_collapse_{label}_persistent"] = float(beta[persistent[0]]) if len(persistent) else np.nan
            row[f"beta_collapse_{label}_persistent_censored"] = not bool(len(persistent))
        aggregate_rows.append(row)
    return pd.concat([seed_rows, pd.DataFrame(aggregate_rows)], ignore_index=True)


def _save_v5_plots(
    successful: pd.DataFrame,
    rd: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create active rate-distortion, collapse, and core-trajectory figures."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 5))
    for (dimension, seed), group in rd.groupby(["latent_dim", "seed"], dropna=False):
        axis.plot(group["posterior_conditional_kl_total_mean"], group["autoencodix_valid_recon_loss"], alpha=.25, label=f"d={dimension}" if seed == rd["seed"].min() else None)
        pareto = group.loc[group["rd_is_pareto_optimal"]]
        axis.scatter(pareto["posterior_conditional_kl_total_mean"], pareto["autoencodix_valid_recon_loss"], s=16)
    axis.set(xlabel="Rate [nats/sample]", ylabel="AUTOENCODIX validation reconstruction loss", title="Rate–distortion trajectories")
    axis.legend()
    figure.savefig(plot_dir / "rate_distortion.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    named_metrics = {
        "posterior_snr": ("posterior_snr_mean",),
        "effective_dimension": ("dimensionality_mu_effective_rank_fraction",),
        "geometry_lcmc": ("geometry_mu_lcmc_zadu_k10", "geometry_mu_lcmc_zadu_k20"),
        "topology": (
            "topology_nts_m",
            "topology_srtd_lite",
            "topology_nts_e",
            "topology_h0_normalized_wasserstein",
            "topology_h1_normalized_wasserstein",
        ),
        "dtm_topology": (
            "dtm_h0_wasserstein",
            "dtm_h1_wasserstein",
                ),
        "collapse": ("posterior_collapsed_dimensions_fraction",),
    }
    positive = successful.loc[pd.to_numeric(successful["beta"], errors="coerce") > 0]
    for filename, metrics in named_metrics.items():
        available = [metric for metric in metrics if metric in positive]
        if not available:
            continue
        figure, axes = plt.subplots(len(available), 1, figsize=(7, 4 * len(available)), squeeze=False)
        plotted = False
        for axis, metric in zip(axes[:, 0], available):
            summary = positive.groupby(["latent_dim", "beta"])[metric].agg(median="median", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75)).reset_index()
            summary = summary.dropna(subset=["beta", "median"])
            if summary.empty:
                axis.set_visible(False)
                continue
            for dimension, group in summary.groupby("latent_dim"):
                beta = group["beta"].to_numpy(float)
                axis.plot(beta, group["median"], marker="o", label=f"d={dimension}")
                axis.fill_between(beta, group["q25"], group["q75"], alpha=.15)
                plotted = True
            _mark_beta_states(axis, successful)
            axis.set_xscale("log")
            axis.set(xlabel="beta (log scale)", ylabel=metric)
            axis.legend()
        if not plotted:
            plt.close(figure)
            continue
        figure.savefig(plot_dir / f"{filename}.png", dpi=200, bbox_inches="tight")
        plt.close(figure)

    robustness_files = sorted(
        output_dir.parent.glob(
            "absolute_core_metrics_v1/dtm_topology/*/dtm_robustness_summary.csv"
        )
    )
    if robustness_files:
        robustness = pd.concat(
            [pd.read_csv(path) for path in robustness_files], ignore_index=True
        )
        run_beta = successful[["run_name", "beta"]].drop_duplicates("run_name")
        robustness = robustness.merge(run_beta, on="run_name", how="left")
        robustness.to_csv(output_dir / "dtm_mass_robustness_summary.csv", index=False)
        figure, axes = plt.subplots(2, 1, figsize=(7, 8), squeeze=False)
        for axis, metric in zip(
            axes[:, 0], ("dtm_h0_wasserstein", "dtm_h1_wasserstein"), strict=True
        ):
            for mass, group in robustness.groupby("dtm_mass_fraction", sort=True):
                summary = group.groupby("beta")[metric].median().reset_index()
                axis.plot(summary["beta"], summary[metric], marker="o", label=f"mass={mass:g}")
            axis.set(xlabel="beta", ylabel=metric)
            axis.legend()
        figure.savefig(plot_dir / "dtm_mass_robustness.png", dpi=200, bbox_inches="tight")
        plt.close(figure)

def save_sweep_analysis(frame: pd.DataFrame, output_dir: Path) -> None:
    """Persist only active rate-distortion and collapse sweep analyses."""
    frame = frame.copy()
    canonical = 'posterior_collapsed_dimensions_fraction_eps_0p01_delta_0p01'
    if canonical in frame:
        frame['posterior_collapsed_dimensions_fraction'] = frame[canonical]
    output_dir.mkdir(parents=True, exist_ok=True)
    if 'autoencodix_valid_recon_loss' not in frame:
        frame['autoencodix_valid_recon_loss'] = np.nan
    successful = frame.loc[frame.get("status", "ok").eq("ok")].copy() if "status" in frame else frame.copy()
    rd = build_rate_distortion_summary(successful)
    collapse = collapse_boundaries(successful)
    rd.to_csv(output_dir / "rate_distortion_summary.csv", index=False)
    collapse.to_csv(output_dir / "collapse_boundary_summary.csv", index=False)
    _save_v5_plots(successful, rd, output_dir)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frames = [pd.read_csv(path) for path in args.input]
    save_sweep_analysis(pd.concat(frames, ignore_index=True, sort=False), args.output_dir)


if __name__ == "__main__":
    main()
