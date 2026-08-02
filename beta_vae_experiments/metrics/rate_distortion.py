"""Cross-run rate-distortion analysis for beta-VAE experiment grids."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from paretoset import paretoset

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt

RATE_COLUMN = "posterior_conditional_kl_total_mean"
DISTORTION_COLUMN = "recon_test_mse"
RATE_DISTORTION_SOURCE = (
    "Alemi, Fischer, Dillon, and Murphy, Fixing a Broken ELBO, "
    "ICML 2018, arXiv:1711.00464."
)


def _pareto_mask(rate: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    """Return the non-dominated rate-distortion set via ``paretoset`` 1.2.5.

    The objective vector for run ``i`` is ``f_i = (R_i, D_i)``, with both
    components minimized. Run ``a`` dominates run ``b`` exactly when

        R_a <= R_b and D_a <= D_b,

    with at least one strict inequality. The Pareto set contains every run for
    which no such dominating run exists. Identical objective vectors therefore
    remain jointly non-dominated.

    The dominance algorithm is delegated to the public
    ``paretoset(..., sense=["min", "min"], distinct=False)`` API. Setting
    ``distinct=False`` preserves every run when multiple runs have identical
    objective vectors. This adapter only fixes those semantics and converts
    the returned mask to a NumPy boolean array.
    """
    objectives = np.column_stack(
        (
            np.asarray(rate, dtype=np.float64),
            np.asarray(distortion, dtype=np.float64),
        )
    )
    return np.asarray(
        paretoset(
            objectives,
            sense=["min", "min"],
            distinct=False,
        ),
        dtype=bool,
    )


def build_rate_distortion_table(overview: pd.DataFrame) -> pd.DataFrame:
    """Build a normalized test-set rate-distortion table across completed runs.

    For the diagonal Gaussian posterior
    ``q(z|x)=N(mu(x), diag(exp(logvar(x))))`` and standard-normal prior
    ``p(z)=N(0,I)``, rate is the analytical conditional KL in nats per sample:

        R = E_x KL(q(z|x) || p(z))
          = E_x 1/2 sum_j [mu_j(x)^2 + exp(logvar_j(x))
                           - 1 - logvar_j(x)].

    Distortion is reconstruction MSE per input element on the fixed test split:

        D = 1/(n*d) sum_i sum_l (x_i,l - x_hat_i,l)^2.

    The two-objective model is ``minimize (R, D)``. Non-dominated membership is
    computed by the public ``paretoset`` 1.2.5 API, not by a project-local
    sorting implementation. Grouping by ``loss_reduction`` and interpreting
    KL-trained rate as a direct objective diagnostic versus MMD-trained rate as
    a post-hoc information diagnostic remain explicit project reporting
    choices.

    ``R`` and ``D`` remain comparable across ``sum`` and ``mean`` training
    reductions because neither reuses the raw reduced training loss.
    """
    required = {
        "run_name",
        "status",
        "loss_reduction",
        "default_vae_loss",
        "learning_rate",
        "beta",
        RATE_COLUMN,
        DISTORTION_COLUMN,
    }
    missing = required.difference(overview.columns)
    if missing:
        raise ValueError(
            "Rate-distortion overview is missing columns: " + ", ".join(sorted(missing))
        )

    columns = [
        "run_name",
        "status",
        "loss_reduction",
        "default_vae_loss",
        "learning_rate",
        "beta",
        "epochs",
        "seed",
        RATE_COLUMN,
        DISTORTION_COLUMN,
    ]
    available = [column for column in columns if column in overview.columns]
    table = overview.loc[overview["status"].eq("ok"), available].copy()
    table = table.loc[
        np.isfinite(pd.to_numeric(table[RATE_COLUMN], errors="coerce"))
        & np.isfinite(pd.to_numeric(table[DISTORTION_COLUMN], errors="coerce"))
    ].copy()
    table["rate_nats_per_sample"] = table[RATE_COLUMN].astype(float)
    table["distortion_mse_per_element"] = table[DISTORTION_COLUMN].astype(float)
    table["scientific_source"] = RATE_DISTORTION_SOURCE
    table["rate_interpretation"] = np.where(
        table["default_vae_loss"].eq("kl"),
        "direct_kl_objective_diagnostic",
        "post_hoc_kl_information_rate",
    )
    table["is_pareto_optimal_within_reduction"] = False
    for indices in table.groupby("loss_reduction", sort=False).groups.values():
        positions = table.index.get_indexer(indices)
        mask = _pareto_mask(
            table.loc[indices, "rate_nats_per_sample"].to_numpy(dtype=float),
            table.loc[indices, "distortion_mse_per_element"].to_numpy(dtype=float),
        )
        table.loc[
            table.index[positions[mask]], "is_pareto_optimal_within_reduction"
        ] = True
    return table.sort_values(
        ["loss_reduction", "default_vae_loss", "learning_rate", "beta"]
    ).reset_index(drop=True)


def _plot_rate_distortion(table: pd.DataFrame, path: Path) -> None:
    reductions = [
        reduction
        for reduction in ("sum", "mean")
        if reduction in set(table["loss_reduction"])
    ]
    if not reductions:
        return
    figure, axes = plt.subplots(
        1,
        len(reductions),
        figsize=(7 * len(reductions), 5),
        squeeze=False,
        sharey=True,
    )
    colors = {"kl": "tab:blue", "mmd": "tab:orange"}
    markers = {0.0001: "o", 0.0005: "s"}
    for axis, reduction in zip(axes[0], reductions):
        subset = table.loc[table["loss_reduction"].eq(reduction)]
        for (loss, learning_rate), group in subset.groupby(
            ["default_vae_loss", "learning_rate"], sort=True
        ):
            ordered = group.sort_values("beta")
            label = f"{loss.upper()}, lr={learning_rate:g}"
            axis.plot(
                ordered["rate_nats_per_sample"],
                ordered["distortion_mse_per_element"],
                color=colors.get(str(loss), "0.4"),
                alpha=0.35,
                linewidth=1,
            )
            points = axis.scatter(
                ordered["rate_nats_per_sample"],
                ordered["distortion_mse_per_element"],
                c=np.log10(ordered["beta"].to_numpy(dtype=float)),
                cmap="viridis",
                marker=markers.get(float(learning_rate), "o"),
                edgecolor=colors.get(str(loss), "0.4"),
                linewidth=1,
                label=label,
            )
        pareto = subset.loc[subset["is_pareto_optimal_within_reduction"]].sort_values(
            "rate_nats_per_sample"
        )
        if not pareto.empty:
            axis.plot(
                pareto["rate_nats_per_sample"],
                pareto["distortion_mse_per_element"],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Pareto front",
            )
            axis.scatter(
                pareto["rate_nats_per_sample"],
                pareto["distortion_mse_per_element"],
                facecolors="none",
                edgecolors="black",
                s=90,
                linewidths=1.5,
            )
        axis.set_title(f"loss_reduction={reduction}")
        axis.set_xlabel("Rate: conditional KL [nats/sample]")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0, 0].set_ylabel("Distortion: test reconstruction MSE/element")
    colorbar = figure.colorbar(points, ax=axes.ravel().tolist(), shrink=0.85)
    colorbar.set_label("log10(beta)")
    figure.suptitle("Rate–distortion comparison across beta-VAE runs")
    figure.subplots_adjust(left=0.08, right=0.9, bottom=0.12, top=0.88, wspace=0.12)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_rate_distortion_analysis(
    overview: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Persist the cross-run rate-distortion table, Pareto labels, and plot."""
    table = build_rate_distortion_table(overview)
    analysis_dir = Path(output_dir) / "rate_distortion"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(analysis_dir / "rate_distortion.csv", index=False)
    try:
        table.to_parquet(analysis_dir / "rate_distortion.parquet", index=False)
    except (ImportError, ModuleNotFoundError):
        pass
    if not table.empty:
        _plot_rate_distortion(table, analysis_dir / "rate_distortion.png")
    return table
