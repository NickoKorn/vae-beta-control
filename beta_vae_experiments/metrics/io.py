"""Persistence for scalar tables and numeric artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import AbsoluteCoreMetricConfig, AbsoluteCoreMetricResults


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_frame(frame: pd.DataFrame, stem: Path, warnings: list[str]) -> None:
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    try:
        frame.to_parquet(stem.with_suffix(".parquet"), index=False)
    except (ImportError, ModuleNotFoundError) as exc:
        warnings.append(f"Parquet output unavailable for {stem.name}: {exc}")


def _sort_result_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic scientific-result ordering where keys exist."""
    preferred = [
        "experiment_family",
        "loss_reduction",
        "default_vae_loss",
        "learning_rate",
        "beta_effective",
        "beta",
        "training_seed",
        "posterior_sample_id",
        "split",
        "metric_role",
        "group",
        "metric_name",
        "run_name",
        "latent_dimension",
    ]
    keys = [column for column in preferred if column in frame.columns]
    return (
        frame.sort_values(keys, kind="stable", na_position="last").reset_index(
            drop=True
        )
        if keys
        else frame.reset_index(drop=True)
    )


def _merge_run_frame(
    new_frame: pd.DataFrame,
    stem: Path,
    run_name: str,
    replace_existing_run: bool,
) -> pd.DataFrame:
    csv_path = stem.with_suffix(".csv")
    if not csv_path.exists():
        return new_frame
    existing = pd.read_csv(csv_path)
    if "run_name" not in existing:
        raise ValueError(f"Existing table {csv_path} has no run_name column")
    already_exists = existing["run_name"].astype(str).eq(str(run_name))
    if already_exists.any() and not replace_existing_run:
        raise FileExistsError(
            f"Run {run_name!r} already exists in {csv_path}; "
            "set replace_existing_run=True to replace it"
        )
    existing = existing.loc[~already_exists]
    return _sort_result_frame(
        pd.concat([existing, new_frame], ignore_index=True, sort=False)
    )


def save_absolute_core_metric_results(
    results: AbsoluteCoreMetricResults,
    config: AbsoluteCoreMetricConfig,
) -> Path:
    output = Path(config.output_dir) / "absolute_core_metrics_v1"
    output.mkdir(parents=True, exist_ok=True)
    wide = _sort_result_frame(
        _merge_run_frame(
            results.scalar_metrics_wide,
            output / "scalar_metrics_wide",
            results.run_name,
            config.replace_existing_run,
        )
    )
    long = _sort_result_frame(
        _merge_run_frame(
            results.scalar_metrics_long,
            output / "scalar_metrics_long",
            results.run_name,
            config.replace_existing_run,
        )
    )
    per_dimension = _sort_result_frame(
        _merge_run_frame(
            results.per_dimension_metrics,
            output / "per_dimension_metrics",
            results.run_name,
            config.replace_existing_run,
        )
    )
    _write_frame(wide, output / "scalar_metrics_wide", results.warnings)
    _write_frame(long, output / "scalar_metrics_long", results.warnings)
    _write_frame(per_dimension, output / "per_dimension_metrics", results.warnings)
    results.metric_catalog.to_csv(output / "metric_catalog.csv", index=False)
    (output / "metric_catalog.json").write_text(
        results.metric_catalog.to_json(orient="records", indent=2),
        encoding="utf-8",
    )
    run_metadata = output / "run_metadata"
    run_metadata.mkdir(parents=True, exist_ok=True)
    (run_metadata / f"{results.run_name}.json").write_text(
        json.dumps(results.metadata, indent=2, default=_json_default),
        encoding="utf-8",
    )

    run_name = results.run_name
    posterior = output / "posterior" / run_name
    dimensionality = output / "dimensionality" / run_name
    geometry = output / "geometry" / run_name
    topology = output / "topology" / run_name
    posterior.mkdir(parents=True, exist_ok=True)
    dimensionality.mkdir(parents=True, exist_ok=True)
    geometry.mkdir(parents=True, exist_ok=True)
    topology.mkdir(parents=True, exist_ok=True)

    tables = results.artifacts.get("tables", {})
    if "collapse_curve" in tables:
        tables["collapse_curve"].to_csv(posterior / "collapse_curve.csv", index=False)
    if "collapse_summary_curve" in tables:
        tables["collapse_summary_curve"].to_csv(
            posterior / "collapse_summary_curve.csv", index=False
        )
    if "geometry_summary" in tables:
        tables["geometry_summary"].to_csv(
            geometry / "geometry_summary.csv", index=False
        )
    if "dimensionality_mu_spectrum" in tables:
        tables["dimensionality_mu_spectrum"].to_csv(
            dimensionality / "mu_spectrum.csv", index=False
        )
    arrays = results.artifacts.get("arrays", {})

    for key in ("input_h0", "input_h1", "latent_h0", "latent_h1"):
        if key in tables:
            tables[key].to_csv(topology / f"{key}.csv", index=False)
        if key in arrays:
            np.save(topology / f"{key}.npy", arrays[key])
    if "betti_curves" in tables:
        tables["betti_curves"].to_csv(topology / "betti_curves.csv", index=False)

    run_warnings = output / "warnings"
    run_warnings.mkdir(parents=True, exist_ok=True)
    (run_warnings / f"{results.run_name}.json").write_text(
        json.dumps(results.warnings, indent=2), encoding="utf-8"
    )
    return output
