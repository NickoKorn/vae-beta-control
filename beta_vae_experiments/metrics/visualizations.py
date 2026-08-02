"""Generate and persist standard AUTOENCODIX VARIX result plots."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt


def _save_evaluation_table(frame: pd.DataFrame, stem: Path) -> None:
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    try:
        frame.to_parquet(stem.with_suffix(".parquet"), index=False)
    except (ImportError, ModuleNotFoundError):
        pass


def save_dimensionwise_cancer_ridgelines(
    varix: Any,
    plot_dir: Path,
    parameter: str = "CANCER_TYPE_ACRONYM",
) -> list[Path]:
    """Save one cancer-subtype ridgeline plot per posterior-mean dimension."""
    result = varix.result
    final_epoch = int(varix.config.epochs) - 1

    def posterior_mean_frame(split: str, epoch: int) -> pd.DataFrame:
        """Retrieve final mu with IDs and dimension names from the same snapshot."""
        template = result.get_latent_df(epoch=epoch, split=split)
        means = result.mus.get(epoch=epoch, split=split)
        if hasattr(means, "detach"):
            means = means.detach().cpu().numpy()
        means = np.asarray(means, dtype=np.float64)
        if means.ndim != 2 or means.shape != template.shape:
            raise ValueError(
                f"Posterior means for {split!r} have shape {means.shape}; "
                f"expected {template.shape}."
            )
        return pd.DataFrame(means, index=template.index, columns=template.columns)

    latent = pd.concat(
        [
            posterior_mean_frame("train", final_epoch),
            posterior_mean_frame("valid", final_epoch),
            posterior_mean_frame("test", -1),
        ],
        axis=0,
    )
    metadata = varix.visualizer._collect_all_metadata(result=result)
    if latent.empty:
        raise ValueError(
            "No final posterior-mean latent representations are available."
        )
    if parameter not in metadata.columns:
        raise KeyError(f"Metadata column {parameter!r} is unavailable.")
    if not latent.index.is_unique:
        raise ValueError("Latent sample IDs are not unique across dataset splits.")

    labels = metadata.reindex(latent.index)[parameter]
    frame = latent.copy()
    frame[parameter] = labels
    frame = frame.loc[
        frame[parameter].notna()
        & ~frame[parameter].astype(str).str.lower().isin({"unknown", "nan"})
    ]
    frame[parameter] = frame[parameter].astype(str)
    subtype_order = frame[parameter].value_counts().sort_index().index.tolist()
    palette = dict(
        zip(subtype_order, sns.color_palette("tab10", n_colors=len(subtype_order)))
    )

    saved: list[Path] = []
    for dimension in latent.columns:
        dimension_frame = frame[[dimension, parameter]].rename(
            columns={dimension: "latent_value"}
        )
        lower, upper = np.quantile(
            dimension_frame["latent_value"].to_numpy(dtype=float), [0.01, 0.99]
        )
        if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
            lower = float(dimension_frame["latent_value"].min())
            upper = float(dimension_frame["latent_value"].max())

        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        grid = sns.FacetGrid(
            dimension_frame,
            row=parameter,
            hue=parameter,
            row_order=subtype_order,
            hue_order=subtype_order,
            palette=palette,
            height=0.75,
            aspect=9,
            xlim=(lower, upper),
        )
        grid.map_dataframe(
            sns.kdeplot,
            x="latent_value",
            fill=True,
            alpha=0.65,
            bw_adjust=0.6,
            clip_on=True,
            warn_singular=False,
            linewidth=1,
        )
        grid.map_dataframe(
            sns.kdeplot,
            x="latent_value",
            color="white",
            linewidth=0.8,
            bw_adjust=0.6,
            clip_on=True,
            warn_singular=False,
        )
        grid.refline(y=0, linewidth=1, linestyle="-", color=None, clip_on=False)
        grid.figure.subplots_adjust(hspace=-0.35, top=0.92)
        grid.set_titles(row_template="{row_name}")
        grid.set(yticks=[], ylabel="", xlabel="Posterior mean")
        grid.despine(bottom=True, left=True)
        grid.figure.suptitle(
            f"{dimension}: posterior-mean distribution by cancer subtype"
        )

        path = plot_dir / f"Ridgeline_{parameter}_{dimension}.png"
        grid.figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(grid.figure)
        saved.append(path.resolve())
    return saved


def save_varix_result_plots(
    varix: Any,
    output_dir: Path,
    run_name: str,
    params: Sequence[str] = ("SEX", "CANCER_TYPE_ACRONYM"),
) -> list[Path]:
    """Run standard show_result/evaluate and save result and loss plots."""
    plot_dir = Path(output_dir) / "plots" / run_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    before = {path.resolve() for path in plot_dir.glob("*.png")}

    varix.show_result(params=list(params))
    varix.visualizer.save_plots(path=str(plot_dir), which="all", format="png")
    save_dimensionwise_cancer_ridgelines(varix, plot_dir)

    evaluation_result = varix.evaluate(params=list(params))
    evaluation = getattr(evaluation_result, "embedding_evaluation", None)
    if isinstance(evaluation, pd.DataFrame) and not evaluation.empty:
        _save_evaluation_table(evaluation, plot_dir / "embedding_evaluation")
    varix.visualizer.save_plots(path=str(plot_dir), which="all", format="png")
    plt.close("all")

    after = {path.resolve() for path in plot_dir.glob("*.png")}
    return sorted(after - before)
