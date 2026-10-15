"""Preflight invariants for the three central experiment configurations."""

import math
from pathlib import Path

import pytest

from beta_vae_experiments.new_beta_varix_experiments import (
    build_training_runs,
    load_experiment_config,
)


EXPERIMENT_ROOT = Path(__file__).parents[1] / "user_interface"
CONFIGS = {
    "tcga": EXPERIMENT_ROOT / "experiments.json#training.tcga",
    "circle": EXPERIMENT_ROOT / "experiments.json#training.circle",
    "swiss_roll": EXPERIMENT_ROOT / "experiments.json#training.swiss_roll",
}


@pytest.mark.parametrize(("dataset_id", "config_path"), CONFIGS.items())
def test_each_dataset_configuration_has_unique_runs(
    dataset_id: str, config_path: Path
) -> None:
    config = load_experiment_config(config_path)
    grid = build_training_runs(config["experiment_name"], config["training"])
    identities = {
        (
            run["loss"], run["learning_rate"], run["beta"],
            run["seed"], run["latent_dim"],
            run["loss_reduction"],
        )
        for run in grid
    }
    assert config["dataset"]["id"] == dataset_id
    training = config["training"]
    expected_count = math.prod(
        len(training[key])
        for key in (
            "betas", "learning_rates", "latent_dims", "seeds",
            "losses", "loss_reductions",
        )
    )
    assert len(grid) == len(identities) == expected_count


def test_synthetic_dataset_options_are_fully_controlled_by_json() -> None:
    circle = load_experiment_config(CONFIGS["circle"])["dataset"]
    swiss_roll = load_experiment_config(CONFIGS["swiss_roll"])["dataset"]
    assert isinstance(circle["options"], dict)
    assert isinstance(swiss_roll["options"], dict)
    assert isinstance(circle["split_seed"], int)
    assert isinstance(swiss_roll["split_seed"], int)
