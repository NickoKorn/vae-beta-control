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
from beta_vae_experiments.metric_pipeline import (
    AbsoluteCoreMetricConfig,
    run_absolute_core_metrics_pipeline,
)
from beta_vae_experiments.metrics.graph.graph_metrics_pipeline import GraphMetricConfig
from beta_vae_experiments.metrics.dimensionality import DimensionalityMetricConfig
from beta_vae_experiments.metrics.preservation import GeometryMetricConfig
from beta_vae_experiments.metrics.graph.neighbor_mixing_pipeline import NeighborMixingConfig
from beta_vae_experiments.metrics.latent_usage import LatentUsageMetricConfig, PosteriorCollapseConfig
from beta_vae_experiments.data_pipeline import (
    build_core_metric_inputs,
    build_data_config,
    prepare_metric_inputs_from_varix_result,
    save_posteriors,
    save_epoch_snapshots,
)
from beta_vae_experiments.datasets import (
    generate_synthetic_dataset,
    is_synthetic_dataset,
)
from beta_vae_experiments.metrics import (
    TopologyMetricConfig,
)
from beta_vae_experiments.metrics.visualizations import save_varix_result_plots
from beta_vae_experiments.metrics.evaluation.downstream import TARGETS, ClassAUCEvaluator

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
JSON_DIR = EXPERIMENT_DIR / "user_interface"

def build_metric_config(settings: dict[str, Any]) -> AbsoluteCoreMetricConfig:
    """Build each metric family's validated configuration from JSON defaults/overrides."""
    from beta_vae_experiments.metrics.evaluation.downstream import DownstreamMetricConfig
    from beta_vae_experiments.topology_annealing import SigmaTopologyConfig
    metric_settings = settings["metrics"]
    if metric_settings.get("encoder_geometry", {}).get("enabled", False):
        raise ValueError("encoder_geometry was removed from the active pipeline")
    latent_usage_settings = metric_settings.get("latent_usage", {})
    graph_settings = dict(metric_settings.get("graph", {}))
    # Historical profiles may retain this now-complementary curvature switch.
    graph_settings.pop("ollivier", None)
    return AbsoluteCoreMetricConfig(
        output_dir=settings["output_dir"],
        downstream=DownstreamMetricConfig(**metric_settings.get("downstream", {})),
        graph=GraphMetricConfig(**graph_settings),
        dimensionality=DimensionalityMetricConfig(**metric_settings.get("dimensionality", {})),
        geometry=GeometryMetricConfig(**{**metric_settings.get("geometry", {}),
            "core_only": metric_settings.get("preservation_only", False)}),
        neighbor_mixing=NeighborMixingConfig(**metric_settings.get("neighbor_mixing", {})),
        preservation_only=metric_settings.get("preservation_only", False),
        latent_usage=LatentUsageMetricConfig(
            **{
                **latent_usage_settings,
                "collapse": PosteriorCollapseConfig(**latent_usage_settings.get("collapse", {})),
            }
        ),
        topology=TopologyMetricConfig(**metric_settings.get("topology", {})),
        sigma_topology=SigmaTopologyConfig(**metric_settings.get("sigma_topology", {})),
    )

def configure_epoch_snapshots(
    settings: dict[str, Any], *, snapshot_enabled: bool | None = None,
    snapshot_every: int | None = None, snapshot_downstream: bool | None = None,
) -> None:
    """Apply CLI overrides and configure native checkpoints for the requested export."""
    snapshots = settings["training"].setdefault("snapshots", {})
    for name, value in (("enabled", snapshot_enabled), ("every", snapshot_every),
                        ("downstream", snapshot_downstream)):
        if value is not None:
            snapshots[name] = value
    validate_snapshot_config(settings["training"])
    if snapshots.get("enabled", False):
        settings["training"]["checkpoint_interval"] = snapshots.get("every", 50)


def validate_snapshot_config(training):
    """Check options for exporting native checkpoint arrays after training."""
    snapshots = training.get("snapshots", {})
    for name, default in (("enabled", False), ("downstream", True)):
        if type(snapshots.get(name, default)) is not bool:
            raise ValueError(f"snapshots.{name} must be boolean")
    every = snapshots.get("every", 50)
    if type(every) is not int or every < 1:
        raise ValueError("Snapshot interval must be a positive integer")


def load_experiment_config(experiment_json: Path) -> dict[str, Any]:
    from beta_vae_experiments.user_interface.config import read_config
    settings, config_directory = read_config(experiment_json)
    if settings.get("require_results_mount") and not Path(settings["require_results_mount"]).is_mount():
        raise FileNotFoundError("Required results drive is not mounted")
    if "results_date" in settings and not settings.get("output_dir"):
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
        checkpoint_interval=int(
            training.get("snapshots", {}).get("every", 50)
            if training.get("snapshots", {}).get("enabled", False)
            else training["checkpoint_interval"]
        ),
        save_memory=False,
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
    beta_schedule: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Train with the native AUTOENCODIX trainer and an optional beta schedule."""
    from autoencodix.utils._losses import VarixLoss
    loss_type = VarixLoss
    if ladder and ladder.get("enabled", False):
        raise ValueError("The retired dimension-ladder backend is unavailable")
    if beta_schedule:
        from beta_vae_experiments.beta_annealing import make_scheduled_loss
        loss_type = make_scheduled_loss(beta_schedule, config)
    varix = acx.Varix(data=fixed_data_splits, config=config, loss_type=loss_type,
                      evaluator=ClassAUCEvaluator)
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
    splits: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Convert one VARIX result and run the central metric pipeline."""
    dataset = settings["dataset"]
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


def validate_experiment_settings(settings: dict[str, Any]) -> None:
    """Check the enabled snapshot and metric configuration before preparing runs."""
    if settings.get("metric_analysis", {}).get("enabled", False):
        raise ValueError("metric_analysis was removed; disable it in the configuration")
    validate_snapshot_config(settings["training"])
    if settings["metrics"]["absolute_core"]["enabled"]:
        build_metric_config(settings)


def create_and_save_training_grid(settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Create the run grid and save the experiment configuration and grid CSV."""
    output_dir = settings["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = build_training_runs(settings["experiment_name"], settings["training"])
    if not runs:
        raise ValueError("The JSON training grid creates no runs.")
    (output_dir / "run_analysis").mkdir(exist_ok=True)
    config_path = output_dir / "experiment_config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(settings, default=str, indent=2) + "\n")
    grid = pd.DataFrame(runs)
    grid["beta_over_d"] = grid.beta / grid.latent_dim
    grid.to_csv(output_dir / "run_analysis" / "training_grid.csv", index=False)
    return runs


def prepare_experiment_data(
    settings: dict[str, Any], first_run: dict[str, Any]
) -> tuple[DataConfig, DatasetContainer]:
    """Build the data configuration and preprocess one shared set of splits."""
    dataset = settings["dataset"]
    dataset_id = str(dataset["id"])
    data = get_synthetic_data(dataset)
    data_config = build_data_config(PROJECT_ROOT, dataset_id, dataset.get("options"))
    split_config = build_varix_config(
        data_config, dataset_id, settings["training"], first_run,
        global_seed=int(dataset["split_seed"]),
    )
    return data_config, prepare_fixed_data_splits(data, split_config)


def save_split_manifest(fixed_data_splits: DatasetContainer, output_dir: Path) -> None:
    """Record split IDs and refuse reuse of an output directory with changed splits."""
    split_ids = {
        name: list(map(str, getattr(fixed_data_splits, name).sample_ids))
        for name in ("train", "valid", "test")
        if getattr(fixed_data_splits, name) is not None
    }
    ids = [sample for values in split_ids.values() for sample in values]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate or overlapping split sample IDs")
    split_path = output_dir / "split.json"
    if split_path.exists() and json.loads(split_path.read_text()) != split_ids:
        raise ValueError("Split IDs/order differ from the saved experiment; refusing resume")
    split_path.write_text(json.dumps(split_ids, indent=2) + "\n")


def load_experiment_progress(
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, set[str], bool]:
    """Load previous results, completed run names and the setting for rerunning them."""
    overview_path = settings["output_dir"] / "experiment_metrics.csv"
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
    finished = (
        set(overview.loc[overview["status"].eq("ok"), "run_name"].astype(str))
        if "status" in overview else set()
    )
    return overview, finished, rerun


def save_run_beta_schedule(
    training: dict[str, Any], config: VarixConfig, run_folder: Path,
) -> None:
    """Save the effective beta schedule when the run uses a custom schedule."""
    if training.get("beta_schedule"):
        from beta_vae_experiments.beta_annealing import schedule_from_settings
        schedule = schedule_from_settings(training["beta_schedule"], config.beta / config.latent_dim)
        pd.DataFrame({
            "epoch": np.arange(schedule.epochs) + 1,
            "beta_over_d": schedule.factors() * config.beta / config.latent_dim,
            "beta_config": schedule.factors() * config.beta,
        }).to_csv(run_folder / "beta_schedule.csv", index=False)


def save_run_loss_history(result: Any, run_folder: Path) -> None:
    """Export native epoch losses and sub-losses in one table."""
    history = {}
    for name, dynamics in [("loss", result.losses), *result.sub_losses.losses()]:
        for epoch, split_values in dynamics.get().items():
            for split, value in split_values.items():
                record = history.setdefault(
                    (int(epoch), split),
                    dict(epoch_index=int(epoch), epoch=int(epoch) + 1, split=split),
                )
                record[name] = float(np.asarray(value).item())
    pd.DataFrame(history.values()).sort_values(["epoch_index", "split"]).to_csv(
        run_folder / "epoch_losses.csv", index=False,
    )


def save_run_outputs(
    varix: Any, result: Any, config: VarixConfig, settings: dict[str, Any],
    run_name: str, run_folder: Path,
) -> dict[str, dict[str, Any]]:
    """Save the beta schedule, losses, final model and posterior arrays in order."""
    save_run_beta_schedule(settings["training"], config, run_folder)
    save_run_loss_history(result, run_folder)
    torch.save(
        {key: value.detach().cpu().clone()
         for key, value in varix._trainer.get_model().state_dict().items()},
        run_folder / "model_final.pt",
    )
    splits = prepare_metric_inputs_from_varix_result(result, varix=varix)
    save_posteriors(splits, settings["output_dir"], run_name, varix)
    return splits


def update_experiment_overview(
    overview: pd.DataFrame, row: pd.DataFrame, run_name: str,
) -> pd.DataFrame:
    """Replace the previous result row for this run in the overview."""
    if not overview.empty:
        overview = overview.loc[overview["run_name"] != run_name]
    overview = pd.concat([overview, row], ignore_index=True, sort=False)
    return overview


def save_run_plots(varix: Any, settings: dict[str, Any], run_name: str) -> None:
    """Save requested plots, reporting plot errors without invalidating run metrics."""
    plots = settings.get("plots", {})
    if plots.get("enabled") and plots.get("per_run"):
        dataset_id = str(settings["dataset"]["id"])
        params = [] if is_synthetic_dataset(dataset_id) else list(
            settings["training"].get("downstream_targets", TARGETS)
        )
        try:
            save_varix_result_plots(varix, settings["output_dir"], run_name, params=params)
        except Exception as error:  # plotting must not invalidate metrics
            print(f"Plot warning for {run_name}: {type(error).__name__}: {error}")


def create_failed_run_record(
    run: dict[str, Any], run_name: str, settings: dict[str, Any], error: Exception,
) -> pd.DataFrame:
    """Record the failed grid point and its traceback so the sweep can continue."""
    print(
        f"Run failed; continuing with the next configuration: {run_name}\n"
        f"{type(error).__name__}: {error}"
    )
    dataset = settings["dataset"]
    return pd.DataFrame([{
        **run,
        "run_name": run_name,
        "experiment_name": settings["experiment_name"],
        "dataset_id": dataset["id"],
        "data_split_seed": int(dataset["split_seed"]),
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }])


def run_experiment(
    experiment_json: Path, *, snapshot_enabled: bool | None = None,
    snapshot_every: int | None = None, snapshot_downstream: bool | None = None,
) -> pd.DataFrame:
    """Prepare shared data, run each configuration, and save its outputs and metrics."""
    settings = load_experiment_config(experiment_json)
    configure_epoch_snapshots(settings, snapshot_enabled=snapshot_enabled,
                             snapshot_every=snapshot_every, snapshot_downstream=snapshot_downstream)
    validate_experiment_settings(settings)
    runs = create_and_save_training_grid(settings)
    data_config, fixed_data_splits = prepare_experiment_data(settings, runs[0])
    output_dir = settings["output_dir"]
    save_split_manifest(fixed_data_splits, output_dir)
    overview, finished, rerun = load_experiment_progress(settings)
    training = settings["training"]
    dataset_id = str(settings["dataset"]["id"])

    for index, run in enumerate(runs, start=1):
        run_name = _run_name(settings["experiment_name"], run)
        if run_name in finished and not rerun:
            print(f"Skipping completed run {index}/{len(runs)}: {run_name}")
            continue
        print(f"Training run {index}/{len(runs)}: {run_name}")
        varix = result = None
        try:
            config = build_varix_config(
                data_config, dataset_id, training, run, global_seed=run["seed"],
            )
            run_folder = output_dir / "run_analysis" / run_name
            run_folder.mkdir(parents=True, exist_ok=True)
            varix, result = run_varix_training(
                fixed_data_splits, config, training.get("ladder"), training.get("beta_schedule"),
            )
            splits = save_run_outputs(varix, result, config, settings, run_name, run_folder)
            snapshots = training.get("snapshots", {})
            if snapshots.get("enabled", False):
                save_epoch_snapshots(
                    varix, run_folder, every=snapshots.get("every", 50),
                    downstream=snapshots.get("downstream", True),
                    beta_schedule=training.get("beta_schedule"),
                )
            if settings["metrics"]["absolute_core"]["enabled"]:
                row = analyse_varix_training(varix, result, run_name, run, settings, splits)
            else:
                row = pd.DataFrame([{**run, "run_name": run_name}])
            row = row.assign(status="ok", error_type=None, error=None)
            overview = update_experiment_overview(overview, row, run_name)
            _save_overview(overview, output_dir)
            save_run_plots(varix, settings, run_name)
        except Exception as error:  # one grid point must not stop the sweep
            failure = create_failed_run_record(run, run_name, settings, error)
            overview = update_experiment_overview(overview, failure, run_name)
            _save_overview(overview, output_dir)
        finally:
            del result, varix
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return overview


def preflight_experiment(
    experiment_json: Path, *, snapshot_enabled: bool | None = None,
    snapshot_every: int | None = None, snapshot_downstream: bool | None = None,
) -> None:
    """Validate one JSON configuration without preprocessing or training."""
    settings = load_experiment_config(experiment_json)
    configure_epoch_snapshots(settings, snapshot_enabled=snapshot_enabled,
                             snapshot_every=snapshot_every, snapshot_downstream=snapshot_downstream)
    dataset = settings["dataset"]
    runs = build_training_runs(settings["experiment_name"], settings["training"])
    if not runs:
        raise ValueError("The JSON training grid creates no runs.")
    build_metric_config(settings)
    if settings.get("metric_analysis", {}).get("enabled", False):
        raise ValueError("metric_analysis was removed; disable it in the configuration")
    validate_snapshot_config(settings["training"])
    data_config = build_data_config(PROJECT_ROOT, str(dataset["id"]), dataset.get("options"))
    config = build_varix_config(data_config, str(dataset["id"]), settings["training"],
                                runs[0], global_seed=runs[0]["seed"])
    if settings["training"].get("beta_schedule"):
        from beta_vae_experiments.beta_annealing import make_scheduled_loss
        make_scheduled_loss(settings["training"]["beta_schedule"], config)
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
    parser.add_argument(
        "--epoch-snapshots", action=argparse.BooleanOptionalAction, default=None,
        help="Export native Train/Validation checkpoint arrays after training; overrides JSON.",
    )
    parser.add_argument(
        "--snapshot-every", type=int, default=None,
        help="Checkpoint/export interval in epochs (JSON training.snapshots.every; default 50).",
    )
    parser.add_argument(
        "--snapshot-downstream", action=argparse.BooleanOptionalAction, default=None,
        help="Evaluate exported Train/Validation snapshots after training; overrides JSON.",
    )
    arguments = parser.parse_args()
    snapshot_options = dict(snapshot_enabled=arguments.epoch_snapshots,
                            snapshot_every=arguments.snapshot_every,
                            snapshot_downstream=arguments.snapshot_downstream)
    if arguments.preflight:
        preflight_experiment(arguments.experiment_json, **snapshot_options)
    else:
        overview = run_experiment(arguments.experiment_json, **snapshot_options)
        if overview.empty or not overview["status"].eq("ok").all():
            raise SystemExit("Experiment contains failed runs; see experiment_metrics.csv")


if __name__ == "__main__":
    main()
