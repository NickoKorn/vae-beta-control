"""Run the fixed-beta VARIX experiment grid and absolute-core metrics.

Scientific definition of the VARIX latent space
------------------------------------------------
The repository-local raw matrices used by this experiment contain:

    RNA:   17,448 numerical feature columns
    METH:   9,829 numerical feature columns
    total: 27,277 numerical Multi-Omics features
    ANNO:      56 separate annotation/clinical columns

``ANNO`` is declared with ``data_type="ANNOTATION"`` and is therefore not an
encoder-input modality. Variables such as ``CANCER_TYPE_ACRONYM`` and ``SEX``
remain Sample-ID-linked metadata for downstream evaluation; excluding them
from ``X`` prevents direct target leakage.

The configured global ``k_filter=2000`` is a total feature budget, not a
per-modality budget. AUTOENCODIX divides it across the two available numerical
modalities. Because 2000 is divisible by two, the final model input retains:

    RNA:  1,000 variance-selected features
    METH: 1,000 variance-selected features
    total: 2,000 scaled numerical input features

Thus every patient is represented before encoding by one concatenated vector

    x_i = [RNA_i,1, ..., RNA_i,1000, METH_i,1, ..., METH_i,1000]
        in R^2000,

not by a three-dimensional tensor. Every matrix entry is one scalar processed
RNA-expression or methylation feature value. The experiment therefore studies
the compression

    27,277 raw numerical features
        -> 2,000 variance-filtered and MINMAX-scaled model-input features
        -> 8 latent coordinates.

For the ``MULTI_BULK`` configuration used here, RNA and DNA-methylation inputs
are integrated by one trained VARIX model. They are not stored as independent
RNA and methylation latent spaces. A direct one-epoch audit of the real input
data confirmed that ``Result.mus`` and ``Result.latentspaces`` return one
``(n_split_samples, latent_dim)`` NumPy array per split, rather than a
modality-keyed dictionary. With ``latent_dim=8`` the observed shapes were:

    train: (2470, 8)
    valid:  (352, 8)
    test:   (707, 8)

Train, validation, and test therefore contain different patients projected by
the same encoder into the same learned eight-dimensional coordinate system.
They are three split-specific matrices, not three separately trained spaces.
Each row is aligned to the identifier at the same position in
``Result.sample_ids``.

The VAE supplies two scientifically distinct representations for every
patient ``x_i``:

    q_phi(z | x_i) = N(mu_i, diag(sigma_i**2))
    z_i = mu_i + sigma_i * epsilon_i,  epsilon_i ~ N(0, I)

``Result.mus`` stores the deterministic posterior center ``mu_i``.
``Result.latentspaces`` stores the realized, decoder-facing draw ``z_i``.
Neither is mislabeled as the other. Geometry preservation is reported
separately for ``X <-> mu`` and ``X <-> z`` using identical sample IDs and
subsampling indices. A difference between their scores quantifies how much
posterior uncertainty and the realized draw alter the geometry of the
posterior centers.

Scientific precedent exists for both choices:

* Yan, Ma, and Yu, bmVAE, Bioinformatics 2023,
  DOI:10.1093/bioinformatics/btac790, explicitly use ``z = mu_z`` as the
  post-training cell representation for clustering.
* Lopez et al., scVI, Nature Methods 2018,
  DOI:10.1038/s41592-018-0229-2; scVI returns the approximate-posterior mean
  by default for dimensionality reduction and offers posterior samples as an
  alternative.
* Hira et al., Scientific Reports 2021,
  DOI:10.1038/s41598-021-85285-4, use sampled
  ``z = mu + sigma * epsilon`` as compressed cancer multi-omics features.
* Jeon et al., ZADU, IEEE VIS 2023,
  DOI:10.1109/VIS54172.2023.00048, provide the label-free embedding-distortion
  metrics used for the input-versus-latent geometry comparisons.

Snapshot and model-selection convention
---------------------------------------
For train/validation, ``TrainingDynamics.get(epoch=-1, split=...)`` resolves
``-1`` to the highest regular training epoch (``-2`` is the second highest,
and so on). Prediction outputs are stored under the literal epoch key ``-1``;
for ``split="test"`` the implementation preferentially returns that stored
test snapshot.
Latents, posterior means, reconstructions, and sample IDs are always requested
with the same relative epoch and split and are then explicitly aligned by ID.
Validation geometry is intended for beta/hyperparameter selection; test
geometry is reserved for evaluation of a configuration selected without using
the test results.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import traceback
import warnings
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch

import autoencodix as acx
from autoencodix.configs.default_config import DataCase
from autoencodix.configs.varix_config import VarixConfig
from autoencodix.data._datasplitter import PairedUnpairedSplitter

try:
    from beta_vae_experiments.data_pipeline import (
        build_core_metric_inputs,
        build_data_config,
        prepare_analysis_data,
        validate_or_create_split_manifest,
    )
    from beta_vae_experiments.metrics import (
        AbsoluteCoreMetricConfig,
        run_absolute_core_metrics_pipeline,
    )
    from beta_vae_experiments.metrics.beta_analysis import (
        save_single_seed_beta_analysis,
    )
    from beta_vae_experiments.metrics.downstream import (
        save_random_forest_cv_evaluation,
    )
    from beta_vae_experiments.metrics.rate_distortion import (
        save_rate_distortion_analysis,
    )
    from beta_vae_experiments.metrics.visualizations import save_varix_result_plots
    from beta_vae_experiments.references import save_research_references
    from beta_vae_experiments.resumable_trainer import ResumableExperimentTrainer
except ModuleNotFoundError:
    from data_pipeline import (  # type: ignore[no-redef]
        build_core_metric_inputs,
        build_data_config,
        prepare_analysis_data,
        validate_or_create_split_manifest,
    )
    from metrics import (  # type: ignore[no-redef]
        AbsoluteCoreMetricConfig,
        run_absolute_core_metrics_pipeline,
    )
    from metrics.beta_analysis import (
        save_single_seed_beta_analysis,  # type: ignore[no-redef]
    )
    from metrics.downstream import (  # type: ignore[no-redef]
        save_random_forest_cv_evaluation,
    )
    from metrics.rate_distortion import (  # type: ignore[no-redef]
        save_rate_distortion_analysis,
    )
    from metrics.visualizations import (  # type: ignore[no-redef]
        save_varix_result_plots,
    )
    from references import save_research_references  # type: ignore[no-redef]
    from resumable_trainer import ResumableExperimentTrainer  # type: ignore[no-redef]


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_SCHEMA_ID = "varix_fixed_beta_d8_absolute_core_metrics_v4_mps"
OUTPUT_DIR = PROJECT_ROOT / "Real_Data_Latent_d8_FixedBeta_AbsoluteCoreMetrics_v4_MPS"
FULL_METRICS_STEM = "fixed_beta_run_metrics_absolute_core_v4_mps"

DATA_SPLIT_SEED = 2
TRAINING_SEEDS = [2]
MAX_EPOCHS = 50
SMOKE_TEST_EPOCHS = 5
CHECKPOINT_INTERVAL = 5
ANNEAL_FUNCTION = "logistic-mid"
LATENT_DIM = 8
BETAS = [
    0.0004,
    0.001,
    0.002,
    0.004,
    0.008,
    0.016,
    0.032,
    0.064,
    0.1,
    0.25,
    0.5,
    0.75,
    1,
    2,
    4,
    6,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    512,
]
BASE_EXPERIMENTS = [
    {"learning_rate": 0.0001, "default_vae_loss": "kl"},
    {"learning_rate": 0.0001, "default_vae_loss": "mmd"},
    {"learning_rate": 0.0005, "default_vae_loss": "kl"},
    {"learning_rate": 0.0005, "default_vae_loss": "mmd"},
]


def enforce_deterministic_pairing_splits() -> None:
    """Make AUTOENCODIX pairing splits independent of Python hash iteration."""

    def deterministic_split_group(
        splitter: PairedUnpairedSplitter,
        ids: Sequence[str],
    ) -> dict[str, np.ndarray]:
        ordered_ids = sorted(ids)
        rng = np.random.default_rng(splitter.config.global_seed)
        shuffled_ids = rng.permutation(ordered_ids)
        n_samples = len(shuffled_ids)
        n_train = int(n_samples * splitter.config.train_ratio)
        n_valid = int(n_samples * splitter.config.valid_ratio)
        return {
            "train": np.asarray(shuffled_ids[:n_train], dtype=object),
            "valid": np.asarray(
                shuffled_ids[n_train : n_train + n_valid],
                dtype=object,
            ),
            "test": np.asarray(shuffled_ids[n_train + n_valid :], dtype=object),
        }

    PairedUnpairedSplitter._split_group = deterministic_split_group


def set_training_seed(seed: int) -> None:
    """Seed model initialization, stochastic sampling, and loader shuffling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def release_accelerator_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _token(value: Any) -> str:
    token = str(value).strip().lower().replace(".", "p")
    return re.sub(r"[^a-z0-9_-]+", "-", token).strip("-")


def run_name_from_parameters(parameters: dict[str, Any]) -> str:
    return (
        f"{_token(parameters['default_vae_loss'])}"
        f"_lr-{_token(parameters['learning_rate'])}"
        f"_beta-{_token(parameters['beta'])}"
        f"_reduction-{_token(parameters['loss_reduction'])}"
        f"_seed-{_token(parameters['seed'])}"
        f"_d-{LATENT_DIM}"
        f"_e-{_token(parameters['epochs'])}"
        f"_ckpt-{CHECKPOINT_INTERVAL}"
        f"_anneal-{_token(ANNEAL_FUNCTION)}"
    )


def build_experiment_grid(
    epochs: int = MAX_EPOCHS,
    loss_reduction: str = "sum",
) -> list[dict[str, Any]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if loss_reduction not in {"sum", "mean"}:
        raise ValueError("loss_reduction must be 'sum' or 'mean'")
    return [
        {
            **base,
            "seed": seed,
            "beta": float(beta),
            "epochs": int(epochs),
            "loss_reduction": loss_reduction,
        }
        for base in BASE_EXPERIMENTS
        for seed in TRAINING_SEEDS
        for beta in BETAS
    ]


def build_smoke_test_experiment(
    epochs: int,
    loss_reduction: str = "sum",
) -> dict[str, Any]:
    if loss_reduction not in {"sum", "mean"}:
        raise ValueError("loss_reduction must be 'sum' or 'mean'")
    return {
        **BASE_EXPERIMENTS[0],
        "seed": TRAINING_SEEDS[0],
        "beta": float(BETAS[0]),
        "epochs": int(epochs),
        "loss_reduction": loss_reduction,
    }


def build_varix_config(parameters: dict[str, Any]) -> VarixConfig:
    enforce_deterministic_pairing_splits()
    data_config = build_data_config(PROJECT_ROOT)
    return VarixConfig(
        data_config=data_config,
        annotation_columns=["CANCER_TYPE_ACRONYM", "SEX"],
        data_case=DataCase.MULTI_BULK,
        epochs=int(parameters["epochs"]),
        checkpoint_interval=CHECKPOINT_INTERVAL,
        anneal_function=ANNEAL_FUNCTION,
        latent_dim=LATENT_DIM,
        loss_reduction=str(parameters["loss_reduction"]),
        scaling="MINMAX",
        k_filter=2000,
        skip_preprocessing=False,
        global_seed=DATA_SPLIT_SEED,
        reproducible=True,
        device="mps",
        n_gpus=1,
        float_precision="32",
        compile_model=False,
        learning_rate=float(parameters["learning_rate"]),
        default_vae_loss=str(parameters["default_vae_loss"]),
        beta=float(parameters["beta"]),
    )


def save_run_config(
    run_name: str,
    parameters: dict[str, Any],
    config: VarixConfig,
    output_dir: Path,
) -> None:
    path = output_dir / "configs" / f"{run_name}_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": run_name,
        "experiment_schema_id": EXPERIMENT_SCHEMA_ID,
        "experiment_parameters": parameters,
        "training_seed": int(parameters["seed"]),
        "effective_varix_config": config.model_dump(mode="json"),
        "metric_pipeline": "absolute_core_metrics_v1",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def analyze_run(
    varix: Any,
    result: Any,
    run_name: str,
    parameters: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    # ``prepare_analysis_data`` asks TrainingDynamics for epoch=-1 separately
    # for train/valid/test. For train/valid this selects the highest regular
    # epoch; for test it preferentially retrieves the literal -1 prediction
    # snapshot. It extracts
    # X, posterior means mu, realized draws z, reconstructions, and sample IDs,
    # then reorders every array to the input sample-ID order. For this VARIX
    # configuration each latent value is a shared integrated Multi-Omics
    # matrix; modality dictionaries remain supported and are rejected if their
    # IDs/shapes cannot be aligned unambiguously.
    splits = prepare_analysis_data(result)
    validate_or_create_split_manifest(splits, output_dir, DATA_SPLIT_SEED)
    inputs = build_core_metric_inputs(
        run_name,
        splits,
        metadata={
            "experiment_schema_id": EXPERIMENT_SCHEMA_ID,
            "snapshot_epoch_index": -1,
            "data_split_seed": DATA_SPLIT_SEED,
            "training_seed": int(parameters["seed"]),
        },
    )
    results = run_absolute_core_metrics_pipeline(
        inputs,
        AbsoluteCoreMetricConfig(output_dir=output_dir),
        varix,
    )
    row = results.scalar_metrics_wide.iloc[0].to_dict()
    posterior_provenance = getattr(result, "posterior_snapshot_provenance", {})
    posterior_metadata = {}
    for split in ("train", "valid", "test"):
        mu_metadata = posterior_provenance.get(split, {}).get("mu", {})
        posterior_metadata[f"mu_{split}_source"] = mu_metadata.get(
            "source",
            "missing",
        )
        posterior_metadata[f"mu_{split}_all_zero"] = bool(
            mu_metadata.get("all_zero", False)
        )
    row.update(
        {
            **parameters,
            **posterior_metadata,
            "run_name": run_name,
            "experiment_schema_id": EXPERIMENT_SCHEMA_ID,
            "status": "ok",
            "data_split_seed": DATA_SPLIT_SEED,
            "training_seed": int(parameters["seed"]),
        }
    )
    return row


def save_overview(rows: list[dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    sort_columns = [
        column
        for column in (
            "default_vae_loss",
            "loss_reduction",
            "learning_rate",
            "beta",
            "seed",
            "run_name",
        )
        if column in frame.columns
    ]
    if sort_columns:
        frame = frame.sort_values(
            sort_columns, kind="stable", na_position="last"
        ).reset_index(drop=True)
    frame.to_csv(output_dir / f"{FULL_METRICS_STEM}.csv", index=False)
    try:
        frame.to_parquet(output_dir / f"{FULL_METRICS_STEM}.parquet", index=False)
    except (ImportError, ModuleNotFoundError) as exc:
        warnings.warn(f"Could not write Parquet overview: {exc}")
    rate_distortion_columns = {
        "recon_test_mse",
        "posterior_conditional_kl_total_mean",
    }
    if rate_distortion_columns.issubset(frame.columns):
        save_rate_distortion_analysis(frame, output_dir)
    beta_analysis_columns = {
        "beta",
        "loss_reduction",
        "default_vae_loss",
        "input_dimension",
        "latent_dimension",
        "recon_test_mse",
        "posterior_conditional_kl_total_mean",
        "info_mutual_information_nats",
        "info_total_correlation_beta_tcvae_mixture_nats",
        "info_dimension_wise_kl_nats",
        "info_decomposition_rate_reference_nats",
        "posterior_collapsed_dimensions_fraction",
    }
    if beta_analysis_columns.issubset(frame.columns):
        save_single_seed_beta_analysis(frame, output_dir)
    return frame


def run_experiment_grid(
    experiments: Sequence[dict[str, Any]],
    output_dir: Path,
    resume_existing: bool = True,
) -> pd.DataFrame:
    save_research_references(output_dir)
    overview_path = output_dir / f"{FULL_METRICS_STEM}.csv"
    rows: list[dict[str, Any]] = []
    if resume_existing and overview_path.exists():
        rows = pd.read_csv(overview_path).to_dict("records")
        for row in rows:
            if pd.isna(row.get("loss_reduction")):
                row["loss_reduction"] = "sum"
    completed = set()
    for row in rows:
        run_name = str(row["run_name"])
        ridgeline_dir = output_dir / "plots" / run_name
        downstream_dir = output_dir / "downstream" / run_name
        dimensionwise_ridges = list(
            ridgeline_dir.glob("Ridgeline_CANCER_TYPE_ACRONYM_LatDim_*.png")
        )
        random_forest_evaluation = (
            downstream_dir / "random_forest_cv5_accuracy_evaluation.csv"
        )
        if (
            str(row.get("status")) == "ok"
            and str(row.get("experiment_schema_id")) == EXPERIMENT_SCHEMA_ID
            and len(dimensionwise_ridges) == LATENT_DIM
            and random_forest_evaluation.exists()
        ):
            completed.add(run_name)
    for parameters in experiments:
        run_name = run_name_from_parameters(parameters)
        if run_name in completed:
            print(f"[SKIP] {run_name}")
            continue
        started_at = datetime.now(UTC)
        started_at_utc = started_at.isoformat(timespec="seconds")
        start_clock = perf_counter()
        print(f"[START {started_at_utc}] {run_name}")
        try:
            set_training_seed(int(parameters["seed"]))
            config = build_varix_config(parameters)
            save_run_config(run_name, parameters, config, output_dir)
            ResumableExperimentTrainer.configure_checkpoint(
                output_dir / "training_checkpoints" / f"{run_name}.pt",
                run_name,
            )
            varix = acx.Varix(config=config)
            varix._trainer_type = ResumableExperimentTrainer
            result = varix.run()
            row = analyze_run(varix, result, run_name, parameters, output_dir)
            try:
                save_varix_result_plots(
                    varix,
                    output_dir=output_dir,
                    run_name=run_name,
                    params=("SEX", "CANCER_TYPE_ACRONYM"),
                )
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                warnings.warn(f"VARIX final show_result/evaluate step failed: {exc}")
            try:
                save_random_forest_cv_evaluation(
                    varix,
                    output_dir=output_dir,
                    run_name=run_name,
                )
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                warnings.warn(f"VARIX downstream Random-Forest step failed: {exc}")
        except torch.cuda.OutOfMemoryError as exc:
            row = {
                **parameters,
                "run_name": run_name,
                "experiment_schema_id": EXPERIMENT_SCHEMA_ID,
                "status": "cuda_oom",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        # A grid run must record an unexpected per-configuration failure and
        # continue with the remaining configurations.
        except Exception as exc:  # noqa: BLE001
            row = {
                **parameters,
                "run_name": run_name,
                "experiment_schema_id": EXPERIMENT_SCHEMA_ID,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            warnings.warn(f"Run failed {run_name}: {exc}")
        finally:
            release_accelerator_memory()
        finished_at_utc = datetime.now(UTC).isoformat(timespec="seconds")
        elapsed_seconds = perf_counter() - start_clock
        row.update(
            {
                "run_started_at_utc": started_at_utc,
                "run_finished_at_utc": finished_at_utc,
                "run_elapsed_seconds": float(elapsed_seconds),
            }
        )
        print(
            f"[FINISH {finished_at_utc}] {run_name} "
            f"status={row['status']} elapsed={elapsed_seconds:.1f}s"
        )
        rows = [existing for existing in rows if existing.get("run_name") != run_name]
        rows.append(row)
        save_overview(rows, output_dir)
    return save_overview(rows, output_dir)


def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a VARIX smoke test and optionally the fixed-beta grid."
    )
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--smoke-test-epochs", type=int, default=SMOKE_TEST_EPOCHS)
    parser.add_argument(
        "--loss-reduction",
        choices=("sum", "mean"),
        default="sum",
        help="PyTorch reduction for reconstruction and variational losses.",
    )
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0 or args.smoke_test_epochs <= 0:
        parser.error("epoch counts must be positive")
    return args


def confirm_full_training(n_runs: int, epochs: int) -> bool:
    if not os.isatty(0):
        print("Full grid was not started; pass --yes after reviewing the smoke test.")
        return False
    answer = input(f"Start {n_runs} full training runs with {epochs} epochs? Type JA: ")
    return answer.strip().upper() == "JA"


def main() -> None:
    cli = parse_command_line()
    if not cli.skip_smoke_test:
        smoke = build_smoke_test_experiment(
            cli.smoke_test_epochs,
            loss_reduction=cli.loss_reduction,
        )
        smoke_dir = OUTPUT_DIR / "smoke_test" / f"epochs_{cli.smoke_test_epochs}"
        overview = run_experiment_grid([smoke], smoke_dir, resume_existing=False)
        if len(overview) != 1 or overview.iloc[0]["status"] != "ok":
            raise RuntimeError(f"Smoke test failed; inspect {smoke_dir}")
    experiments = build_experiment_grid(
        cli.epochs,
        loss_reduction=cli.loss_reduction,
    )
    if cli.yes or confirm_full_training(len(experiments), cli.epochs):
        overview = run_experiment_grid(experiments, OUTPUT_DIR / f"epochs_{cli.epochs}")
        print(overview.head())


if __name__ == "__main__":
    main()
