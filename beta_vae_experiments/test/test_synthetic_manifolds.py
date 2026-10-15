"""Ground-truth manifold dataset and shared-pipeline integration tests."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import autoencodix as acx
import numpy as np
import pandas as pd
import pytest

from beta_vae_experiments.data_pipeline import build_data_config
from beta_vae_experiments.new_beta_varix_experiments import build_varix_config
from beta_vae_experiments.datasets import save_synthetic_ground_truth
from beta_vae_experiments.datasets.synthetic import (
    SYNTHETIC_DATASET_GENERATORS,
    generate_synthetic_dataset,
    is_synthetic_dataset,
    synthetic_split_indices,
)
from beta_vae_experiments.metrics.evaluation.downstream import (
    run_downstream_metrics_pipeline,
)


def test_json_dataset_ids_dispatch_directly_to_registered_generators() -> None:
    assert set(SYNTHETIC_DATASET_GENERATORS) == {"circle", "sphere", "swiss_roll"}
    assert is_synthetic_dataset("circle")
    assert is_synthetic_dataset("sphere")
    assert is_synthetic_dataset("swiss_roll")
    assert not is_synthetic_dataset("tcga")
    with pytest.raises(ValueError, match="Unknown synthetic dataset_id"):
        generate_synthetic_dataset("unknown")


def test_circle_shape_ids_ground_truth_and_seed_reproducibility() -> None:
    config = {"n_samples": 120, "dataset_seed": 42, "split_seed": 17, "radius": 1.0}
    first = generate_synthetic_dataset("circle", config)
    repeated = generate_synthetic_dataset("circle", config)
    changed = generate_synthetic_dataset(
        "circle", {**config, "dataset_seed": 43}
    )

    assert first.x_raw.shape == (120, 2)
    assert first.x_raw.index.is_unique
    assert first.x_raw.index[0] == "circle_000000"
    np.testing.assert_array_equal(first.x_raw, repeated.x_raw)
    assert not np.array_equal(first.x_raw, changed.x_raw)
    np.testing.assert_allclose(
        np.linalg.norm(first.x_raw.to_numpy(), axis=1), 1.0, atol=1e-12
    )
    assert first.metadata["intrinsic_dimension"] == 1
    assert first.metadata["known_betti_0"] == 1
    assert first.metadata["known_betti_1"] == 1


def test_sphere_shape_angles_ground_truth_and_seed_reproducibility() -> None:
    config = {"n_samples": 120, "dataset_seed": 42, "split_seed": 17, "radius": 1.0}
    first = generate_synthetic_dataset("sphere", config)
    repeated = generate_synthetic_dataset("sphere", config)
    changed = generate_synthetic_dataset(
        "sphere", {**config, "dataset_seed": 43}
    )

    assert first.x_raw.shape == (120, 3)
    assert first.x_raw.index.is_unique
    assert first.x_raw.index[0] == "sphere_000000"
    assert first.x_raw.index.equals(first.sample_metadata.index)
    np.testing.assert_array_equal(first.x_raw, repeated.x_raw)
    assert not np.array_equal(first.x_raw, changed.x_raw)
    np.testing.assert_allclose(
        np.linalg.norm(first.x_raw.to_numpy(), axis=1), 1.0, atol=1e-12
    )
    polar = first.sample_metadata["ground_truth_polar"].to_numpy()
    azimuth = first.sample_metadata["ground_truth_azimuth"].to_numpy()
    rebuilt = np.column_stack(
        [
            np.sin(polar) * np.cos(azimuth),
            np.sin(polar) * np.sin(azimuth),
            np.cos(polar),
        ]
    )
    np.testing.assert_allclose(rebuilt, first.x_raw.to_numpy(), atol=1e-10)
    assert first.metadata["intrinsic_dimension"] == 2
    assert first.metadata["known_betti_0"] == 1
    assert first.metadata["known_betti_1"] == 0
    assert first.metadata["known_betti_2"] == 1


def test_swiss_roll_shape_t_alignment_and_seed_reproducibility() -> None:
    config = {"n_samples": 120, "dataset_seed": 42, "split_seed": 17, "noise": 0.0}
    first = generate_synthetic_dataset("swiss_roll", config)
    repeated = generate_synthetic_dataset("swiss_roll", config)
    changed = generate_synthetic_dataset(
        "swiss_roll", {**config, "dataset_seed": 43}
    )

    assert first.x_raw.shape == (120, 3)
    assert len(first.sample_metadata["ground_truth_t"]) == 120
    assert first.x_raw.index.equals(first.sample_metadata.index)
    assert first.x_raw.index.is_unique
    assert first.x_raw.index[0] == "swiss_roll_000000"
    np.testing.assert_array_equal(first.x_raw, repeated.x_raw)
    np.testing.assert_array_equal(first.sample_metadata, repeated.sample_metadata)
    assert not np.array_equal(first.x_raw, changed.x_raw)


def test_synthetic_split_seed_is_deterministic_disjoint_and_aligned() -> None:
    dataset = generate_synthetic_dataset(
        "swiss_roll", {"n_samples": 100, "seed": 42}
    )
    first = synthetic_split_indices(dataset.x_raw.index.tolist(), split_seed=31)
    repeated = synthetic_split_indices(dataset.x_raw.index.tolist(), split_seed=31)
    changed = synthetic_split_indices(dataset.x_raw.index.tolist(), split_seed=32)

    for split in ("train", "valid", "test"):
        np.testing.assert_array_equal(first[split], repeated[split])
    assert not np.array_equal(first["train"], changed["train"])
    sets = {name: set(indices.tolist()) for name, indices in first.items()}
    assert sets["train"].isdisjoint(sets["valid"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["valid"].isdisjoint(sets["test"])
    test_ids = dataset.x_raw.index[first["test"]]
    assert dataset.sample_metadata.loc[test_ids].index.equals(test_ids)
    fake_mu_test = np.zeros((len(test_ids), 4))
    assert dataset.x_raw.loc[test_ids].shape[0] == fake_mu_test.shape[0]


def test_autoencodix_preprocessing_preserves_synthetic_ids_t_and_model_input() -> None:
    options = {"n_samples": 100, "dataset_seed": 42, "split_seed": 19, "noise": 0.0}
    dataset = generate_synthetic_dataset("swiss_roll", options)
    training = {
        "epochs": 1,
        "device": "cpu",
        "checkpoint_interval": 5,
        "anneal_function": "logistic-mid",
    }
    run = {
        "latent_dim": 4, "loss_reduction": "mean", "seed": 2,
        "learning_rate": 5e-4, "loss": "kl", "beta": 0.0,
    }
    project_root = Path(__file__).parents[2]
    pipeline = acx.Varix(
        data=dataset.to_data_package(),
        config=build_varix_config(
            build_data_config(project_root, "swiss_roll"),
            "swiss_roll",
            training,
            run,
            global_seed=19,
        ),
    )
    pipeline.preprocess()

    seen: set[str] = set()
    for split in ("train", "valid", "test"):
        split_dataset = getattr(pipeline.result.datasets, split)
        ids = pd.Index([str(value) for value in split_dataset.sample_ids])
        assert seen.isdisjoint(ids)
        seen.update(ids)
        assert split_dataset.data.shape[0] == len(ids)
        split_metadata = split_dataset.metadata["paired"]
        assert split_metadata.index.astype(str).equals(ids)
        np.testing.assert_array_equal(
            split_metadata["ground_truth_t"].to_numpy(),
            dataset.sample_metadata.loc[ids, "ground_truth_t"].to_numpy(),
        )
    assert len(seen) == 100


def test_clinical_downstream_is_not_applicable_to_synthetic_manifolds() -> None:
    result = run_downstream_metrics_pipeline(
        SimpleNamespace(),
        "synthetic-run",
        dataset_family="synthetic_manifold",
    )
    assert result.empty
    assert list(result.columns) == [
        "metric_name", "value", "status", "warning", "n_samples", "split"
    ]


def test_swiss_roll_ground_truth_artifact_preserves_split_alignment(tmp_path) -> None:
    dataset = generate_synthetic_dataset(
        "swiss_roll", {"n_samples": 100, "seed": 42, "split_seed": 9}
    )
    indices = synthetic_split_indices(dataset.x_raw.index.tolist(), split_seed=9)
    splits = {}
    for split, positions in indices.items():
        ids = dataset.x_raw.index[positions].to_numpy(dtype=object)
        splits[split] = {
            "sample_ids": ids,
            "sample_metadata": dataset.sample_metadata.loc[ids],
        }
    path = save_synthetic_ground_truth(splits, tmp_path, "run")
    assert path is not None
    artifact = pd.read_csv(path)
    assert list(artifact.columns) == ["sample_id", "split", "ground_truth_t"]
    assert len(artifact) == 100
    for row in artifact.itertuples(index=False):
        assert np.isclose(
            row.ground_truth_t,
            dataset.sample_metadata.loc[row.sample_id, "ground_truth_t"],
        )
