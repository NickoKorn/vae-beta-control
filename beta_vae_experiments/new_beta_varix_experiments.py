"""Run one completely JSON-controlled VARIX experiment."""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import re
import traceback
from pathlib import Path
from typing import Any

# Required by torch deterministic CUDA mode before the first CuBLAS operation.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import autoencodix as acx
from autoencodix.configs.default_config import DataCase, DataConfig
from autoencodix.configs.varix_config import VarixConfig
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data.datapackage import DataPackage
import pandas as pd
import numpy as np
import torch
from beta_vae_experiments.run_artifacts import (
    RecordedVarixLoss, save_epoch_history, save_final_posteriors,
    initialize_artifact_layout, verify_split_manifest,
)

from beta_vae_experiments.metric_pipeline import (
    AbsoluteCoreMetricConfig,
    run_absolute_core_metrics_pipeline,
)
from beta_vae_experiments.metrics.graph.graph_metrics_pipeline import GraphMetricConfig
from beta_vae_experiments.metrics.dimensionality import DimensionalityMetricConfig
from beta_vae_experiments.metrics.geometry import GeometryMetricConfig
from beta_vae_experiments.metrics.latent_usage import LatentUsageMetricConfig, PosteriorCollapseConfig
from beta_vae_experiments.data_pipeline import (
    build_core_metric_inputs,
    build_data_config,
    prepare_metric_inputs_from_varix_result,
    save_test_posterior,
    verify_test_posterior,
)
from beta_vae_experiments.datasets import (
    generate_synthetic_dataset,
    is_synthetic_dataset,
)
from beta_vae_experiments.metrics import (
    TopologyMetricConfig,
    LocalTopologyMetricConfig,
)
from beta_vae_experiments.metrics.visualizations import save_varix_result_plots
from beta_vae_experiments.result_analysis.sweep_analysis import save_sweep_analysis
from beta_vae_experiments.metrics.evaluation.downstream import TARGETS, ClassAUCEvaluator

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
JSON_DIR = EXPERIMENT_DIR / "user_interface"

METRIC_SCHEMA_VERSION = "v23"

def build_metric_config(settings: dict[str, Any]) -> AbsoluteCoreMetricConfig:
    """Build each metric family's validated configuration from JSON defaults/overrides."""
    from beta_vae_experiments.metrics.curvature.encoder_geometry import EncoderGeometryConfig
    metric_settings = settings["metrics"]
    latent_usage_settings = metric_settings.get("latent_usage", {})
    return AbsoluteCoreMetricConfig(
        output_dir=settings["output_dir"],
        graph=GraphMetricConfig(**metric_settings.get("graph", {})),
        dimensionality=DimensionalityMetricConfig(**metric_settings.get("dimensionality", {})),
        geometry=GeometryMetricConfig(**metric_settings.get("geometry", {})),
        encoder_geometry=EncoderGeometryConfig(**metric_settings.get("encoder_geometry", {})),
        latent_usage=LatentUsageMetricConfig(
            **{
                **latent_usage_settings,
                "collapse": PosteriorCollapseConfig(**latent_usage_settings.get("collapse", {})),
            }
        ),
        topology=TopologyMetricConfig(**metric_settings.get("topology", {})),
        local_topology=LocalTopologyMetricConfig(**metric_settings.get("local_topology", {})),
    )

def load_experiment_config(experiment_json: Path) -> dict[str, Any]:
    from beta_vae_experiments.user_interface.config import read_config
    settings, config_directory = read_config(experiment_json)
    if settings.get("require_results_mount") and not Path(settings["require_results_mount"]).is_mount():
        raise FileNotFoundError("Required results drive is not mounted")
    if "results_date" in settings:
        import datetime
        date = datetime.date.fromisoformat(settings["results_date"]).isoformat()
        name = settings["experiment_name"]
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("experiment_name must be a directory name")
        results_root = Path(settings.get("results_root", EXPERIMENT_DIR))
        if not results_root.is_absolute():
            results_root = (config_directory / results_root).resolve()
        settings["output_dir"] = str(results_root / f"new_results_{date}" / name)
    output_dir = Path(settings["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (config_directory / output_dir).resolve()
    settings["output_dir"] = output_dir
    return settings

def get_synthetic_data(dataset_settings: dict[str, Any]) -> DataPackage | None:
    """Return direct synthetic input; biology is loaded through DataConfig."""
    dataset_id = str(dataset_settings["id"])
    if dataset_id == "tcga" and os.environ.get("PYTHONHASHSEED") != "2":
        raise ValueError("Start TCGA experiments with PYTHONHASHSEED=2 before launching Python.")
    if not is_synthetic_dataset(dataset_id):
        return None
    return generate_synthetic_dataset(
        dataset_id, dataset_settings.get("options", {})
    ).to_data_package()


def build_training_runs(
    experiment_name: str,
    training: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the training grid and discard duplicate generated run names."""
    runs_by_name: dict[str, dict[str, Any]] = {}
    normalized_beta = "beta_over_d" in training
    if ("betas" in training) == normalized_beta:
        raise ValueError("Specify exactly one of betas or beta_over_d.")
    if normalized_beta and (
        training["losses"] != ["kl"] or training["loss_reductions"] != ["mean"]
    ):
        raise ValueError("beta_over_d requires KL loss with mean reduction.")
    beta_grid = training["beta_over_d"] if normalized_beta else training["betas"]

    for beta, learning_rate, latent_dim, seed, loss, loss_reduction in itertools.product(
        beta_grid,
        training["learning_rates"],
        training["latent_dims"],
        training["seeds"],
        training["losses"],
        training["loss_reductions"],
    ):
        # Mean KL divides total rate by d; scale beta to keep its coefficient fixed.
        beta = float(beta) * int(latent_dim) if normalized_beta else float(beta)
        run_config = {
            k: v for k, v in training.items()
            if not isinstance(v, (list, dict))
        }

        run_config.update({
            "beta": float(beta),
            "learning_rate": float(learning_rate),
            "latent_dim": int(latent_dim),
            "seed": int(seed),
            "loss": str(loss),
            "loss_reduction": str(loss_reduction),
        })
        if training.get("ladder", {}).get("enabled", False):
            raise ValueError("The retired dimension-ladder backend is unavailable; use standard beta or beta_schedule")
        if training.get("beta_schedule"):
            from beta_vae_experiments.beta_annealing import schedule_from_settings, schedule_tag
            if not normalized_beta or training["anneal_function"] == "no-annealing":
                raise ValueError("beta_schedule requires beta_over_d and an enabled native annealing adapter")
            schedule = schedule_from_settings(training["beta_schedule"], beta / latent_dim)
            if int(training["epochs"]) != schedule.epochs:
                raise ValueError("training.epochs differs from beta_schedule duration")
            run_config["schedule"] = schedule_tag(training["beta_schedule"])

        if training.get("geometry_intervention"):
            raise ValueError("Neighborhood intervention is archived; use beta_schedule")
        run_name = _run_name(experiment_name, run_config)
        runs_by_name.setdefault(run_name, run_config)

    return list(runs_by_name.values())


def build_varix_config(
    data_config: DataConfig,
    dataset_id: str,
    training: dict[str, Any],
    run: dict[str, Any],
    *,
    global_seed: int,
) -> VarixConfig:
    """Build one VARIX configuration from JSON and one training run."""
    is_synthetic = is_synthetic_dataset(dataset_id)
    return VarixConfig(
        data_config=data_config,
        annotation_columns=([] if is_synthetic else list(training.get("downstream_targets", TARGETS))),
        data_case=DataCase.MULTI_BULK,
        epochs=int(training["epochs"]),
        checkpoint_interval=int(training["checkpoint_interval"]),
        anneal_function=str(training["anneal_function"]),
        latent_dim=int(run["latent_dim"]),
        loss_reduction=str(run["loss_reduction"]),
        scaling="MINMAX", k_filter=(None if is_synthetic else 2000),
        skip_preprocessing=False, global_seed=int(global_seed), reproducible=True,
        device=str(training["device"]), n_gpus=1, float_precision="32",
        compile_model=False, learning_rate=float(run["learning_rate"]),
        default_vae_loss=str(run["loss"]), beta=float(run["beta"]),
    )


def prepare_fixed_data_splits(
    data: DataPackage | None, config: VarixConfig
) -> DatasetContainer:
    """Let AUTOENCODIX preprocess and split the dataset once."""
    preprocessing_varix = acx.Varix(data=data, config=config)
    preprocessing_varix.preprocess()
    if preprocessing_varix.preprocessed_data is None:
        raise ValueError("AUTOENCODIX did not create preprocessed data splits.")
    return preprocessing_varix.preprocessed_data


def run_varix_training(
    fixed_data_splits: DatasetContainer, config: VarixConfig,
    ladder: dict[str, Any] | None = None,
    snapshots: dict[str, Any] | None = None,
    run_folder: Path | None = None,
    beta_schedule: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Train VARIX with the same precomputed splits used by every run.

    Standard training retains RecordedVarixLoss. Optional snapshots use
    mode="all_train_valid"; absent/disabled retains native GeneralTrainer.
    The retired dimension-ladder backend is rejected explicitly.
    """
    loss_type = RecordedVarixLoss
    if ladder and ladder.get("enabled", False):
        raise ValueError("The retired dimension-ladder backend is unavailable")
    if beta_schedule:
        from beta_vae_experiments.beta_annealing import make_scheduled_loss
        loss_type = make_scheduled_loss(beta_schedule, config)
    extra = {}
    if snapshots and snapshots.get("enabled", False):
        if run_folder is None:
            raise ValueError("snapshots need the run folder")
        if snapshots.get("mode") == "all_train_valid":
            from beta_vae_experiments.epoch_snapshot_trainer import make_epoch_snapshot_trainer
            extra["trainer_type"] = make_epoch_snapshot_trainer(
                run_folder, int(snapshots.get("every", 100)), snapshots.get("save_models", False))
        else:
            raise ValueError('Unsupported snapshot mode; use mode="all_train_valid"')
    varix = acx.Varix(data=fixed_data_splits, config=config, loss_type=loss_type,
                      evaluator=ClassAUCEvaluator, **extra)
    return varix, varix.run()


def _run_name(experiment_name: str, run: dict[str, Any]) -> str:
    values = "_".join(f"{key}-{value}" for key, value in run.items())
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{experiment_name}_{values}")
    if len(name) > 220:
        import hashlib
        name = name[:190] + "-" + hashlib.sha256(name.encode()).hexdigest()[:16]
    return name


def analyse_varix_training(
    varix: Any,
    result: Any,
    run_name: str,
    run: dict[str, Any],
    settings: dict[str, Any],
) -> pd.DataFrame:
    """Convert one VARIX result and run the central metric pipeline."""
    dataset = settings["dataset"]
    splits = prepare_metric_inputs_from_varix_result(result, varix=varix)
    verification = verify_test_posterior(splits, varix)
    save_test_posterior(splits, settings["output_dir"], run_name, verification)
    inputs = build_core_metric_inputs(
        run_name,
        splits,
        metadata={
            "dataset_id": dataset["id"],
            "dataset_family": (
                "synthetic_manifold"
                if is_synthetic_dataset(str(dataset["id"]))
                else "biology"
            ),
            "data_split_seed": int(dataset["split_seed"]),
            "downstream_targets": settings["training"].get("downstream_targets", TARGETS),
            "mu_test_source": splits["test"]["mus_source"],
            "logvar_test_source": splits["test"]["sigmas_source"],
            **run,
        },
    )
    metrics = run_absolute_core_metrics_pipeline(
        inputs,
        build_metric_config(settings),
        varix,
    )
    run_overview = metrics.pivot(
        index="run_name", columns="metric_name", values="value"
    ).reset_index()
    run_overview.columns.name = None
    return run_overview.assign(
        experiment_name=settings["experiment_name"],
        dataset_id=dataset["id"], data_split_seed=int(dataset["split_seed"]),
        metric_schema_version=METRIC_SCHEMA_VERSION,
        **run,
    )


def _save_overview(frame: pd.DataFrame, output_dir: Path) -> Path:
    if "run_name" not in frame:
        raise ValueError("Experiment overview must contain run_name")
    if frame["run_name"].isna().any() or frame["run_name"].duplicated().any():
        raise ValueError("Experiment overview requires unique non-null run_name values")
    path = output_dir / "experiment_metrics.csv"
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    return path


def run_experiment(experiment_json: Path) -> pd.DataFrame:
    """Run fixed-split training, per-run metrics, and final metric analysis."""
    settings = load_experiment_config(experiment_json)
    output_dir = settings["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_settings = settings["dataset"]
    training = settings["training"]
    dataset_id = str(dataset_settings["id"])
    runs = build_training_runs(settings["experiment_name"], training)
    if not runs:
        raise ValueError("The JSON training grid creates no runs.")
    initialize_artifact_layout(output_dir, settings, runs)

    data = get_synthetic_data(dataset_settings)
    data_config = build_data_config(PROJECT_ROOT, dataset_id, dataset_settings.get("options"))
    split_config = build_varix_config(
        data_config, dataset_id, training, runs[0],
        global_seed=int(dataset_settings["split_seed"]),
    )
    fixed_data_splits = prepare_fixed_data_splits(data, split_config)
    verify_split_manifest(fixed_data_splits, output_dir)
    overview_path = output_dir / "experiment_metrics.csv"
    execution = settings.get("execution", {})
    resume = bool(execution.get("resume_existing", False))
    rerun = bool(execution.get("rerun_passed", False))
    overview = (
        pd.read_csv(overview_path)
        if resume and overview_path.is_file() else pd.DataFrame()
    )
    if not overview.empty:
        if "run_name" not in overview:
            raise ValueError("Stored experiment overview has no run_name column")
        if overview["run_name"].isna().any() or overview["run_name"].duplicated().any():
            raise ValueError(
                "Stored experiment overview contains duplicate or null run_name values"
            )
    if "metric_schema_version" in overview:
        current_rows = overview["metric_schema_version"].eq(METRIC_SCHEMA_VERSION)
        if "status" in overview:
            current_rows &= overview["status"].eq("ok")
        finished = set(overview.loc[current_rows, "run_name"].astype(str))
    else:
        finished = set()

    for index, run in enumerate(runs, start=1):
        run_name = _run_name(settings["experiment_name"], run)
        if run_name in finished and not rerun:
            print(f"Skipping completed run {index}/{len(runs)}: {run_name}")
            continue
        print(f"Training run {index}/{len(runs)}: {run_name}")
        varix = result = None
        try:
            config = build_varix_config(
                data_config, dataset_id, training, run, global_seed=run["seed"]
            )
            run_folder = output_dir / "run_analysis" / run_name
            varix, result = run_varix_training(
                fixed_data_splits, config, training.get("ladder"),
                training.get("snapshots"), run_folder, training.get("beta_schedule"))
            if training.get("beta_schedule"):
                from beta_vae_experiments.beta_annealing import schedule_from_settings
                schedule = schedule_from_settings(training["beta_schedule"], config.beta / config.latent_dim)
                pd.DataFrame({"epoch": np.arange(schedule.epochs) + 1,
                              "beta_over_d": schedule.factors() * config.beta / config.latent_dim,
                              "beta_config": schedule.factors() * config.beta}).to_csv(
                                  run_folder / "beta_schedule.csv", index=False)
            save_epoch_history(result, run_folder)
            save_final_posteriors(varix, result, run_folder)
            if training.get("beta_schedule") and training.get("snapshots", {}).get("mode") == "all_train_valid":
                from beta_vae_experiments.result_analysis.analyse_annealing_snapshots import analyse
                analyse(run_folder, k_values=settings.get("recovery_analysis", {}).get("k_values", [5, 10, 20]),
                        max_samples=int(settings.get("recovery_analysis", {}).get("observer_max_samples", 512)),
                        targets=training.get("downstream_targets", []),
                        rate_thresholds=settings.get("recovery_analysis", {}).get("rate_thresholds_nats_per_sample", [.001, .01, .1]))

            if settings["metrics"]["absolute_core"]["enabled"]:
                row = analyse_varix_training(varix, result, run_name, run, settings)
            else:
                row = pd.DataFrame([{**run, "run_name": run_name,
                                     "metric_schema_version": METRIC_SCHEMA_VERSION}])
            row = row.assign(status="ok", error_type=None, error=None)
            if not overview.empty:
                overview = overview.loc[overview["run_name"] != run_name]
            overview = pd.concat([overview, row], ignore_index=True, sort=False)
            _save_overview(overview, output_dir)

            plots = settings.get("plots", {})
            if plots.get("enabled") and plots.get("per_run"):
                params = [] if is_synthetic_dataset(dataset_id) else list(
                    training.get("downstream_targets", TARGETS)
                )
                try:
                    save_varix_result_plots(
                        varix, output_dir, run_name, params=params
                    )
                except Exception as error:  # plotting must not invalidate metrics
                    print(
                        f"Plot warning for {run_name}: "
                        f"{type(error).__name__}: {error}"
                    )
        except Exception as error:  # one grid point must not stop the sweep
            print(
                f"Run failed; continuing with the next configuration: {run_name}\n"
                f"{type(error).__name__}: {error}"
            )
            failure = pd.DataFrame([{
                **run,
                "run_name": run_name,
                "experiment_name": settings["experiment_name"],
                "dataset_id": dataset_settings["id"],
                "data_split_seed": int(dataset_settings["split_seed"]),
                "metric_schema_version": METRIC_SCHEMA_VERSION,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }])
            if not overview.empty:
                overview = overview.loc[overview["run_name"] != run_name]
            overview = pd.concat([overview, failure], ignore_index=True, sort=False)
            _save_overview(overview, output_dir)
        finally:
            del result, varix
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if settings.get("metric_analysis", {}).get("enabled") and not overview.empty:
        analysis_dir = output_dir / "metric_analysis"
        save_sweep_analysis(overview, analysis_dir)
    return overview


def preflight_experiment(experiment_json: Path) -> None:
    """Validate one JSON configuration without preprocessing or training."""
    settings = load_experiment_config(experiment_json)
    dataset = settings["dataset"]
    runs = build_training_runs(settings["experiment_name"], settings["training"])
    if not runs:
        raise ValueError("The JSON training grid creates no runs.")
    build_data_config(PROJECT_ROOT, str(dataset["id"]), dataset.get("options"))
    print(
        f"Preflight passed: {settings['experiment_name']} "
        f"({dataset['id']}, {len(runs)} runs)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one JSON-controlled VARIX experiment."
    )
    parser.add_argument(
        "experiment_json",
        nargs="?",
        type=Path,
        default=JSON_DIR / "experiments.json",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate the configuration without training.",
    )
    arguments = parser.parse_args()
    if arguments.preflight:
        preflight_experiment(arguments.experiment_json)
    else:
        run_experiment(arguments.experiment_json)


if __name__ == "__main__":
    main()
