from pathlib import Path
import math

import pytest

from beta_vae_experiments.new_beta_varix_experiments import (
    _save_overview,
    build_metric_config,
    build_training_runs,
    load_experiment_config,
)
import pandas as pd


CONFIG_ROOT = Path(__file__).parents[1] / "user_interface"
CONFIG_PATH = CONFIG_ROOT / "experiments.json#training.tcga"
EXPERIMENT_CONFIGS = {
    "tcga": CONFIG_PATH,
    "circle": CONFIG_ROOT / "experiments.json#training.circle",
    "swiss_roll": CONFIG_ROOT / "experiments.json#training.swiss_roll",
}


def test_metric_config_defaults_and_json_overrides_are_isolated(tmp_path: Path) -> None:
    defaults = build_metric_config({"output_dir": tmp_path, "metrics": {}})
    configured = build_metric_config({
        "output_dir": tmp_path,
        "metrics": {
            "graph": {"k_values": [5, 10]},
            "dimensionality": {"mle_n_neighbors": 12},
            "downstream": {"n_draws": 3, "sampling_seed": 17},
            "geometry": {"selected_k": 10},
            "latent_usage": {"collapse": {"delta": 0.05, "epsilon_values": [0.01, 0.1]}},
            "topology": {"max_samples": 100},
        },
    })
    assert configured.graph.k_values == (5, 10)
    assert configured.dimensionality.mle_n_neighbors == 12
    assert configured.downstream.n_draws == 3
    assert configured.downstream.sampling_seed == 17
    assert defaults.downstream.n_draws == 10
    assert configured.geometry.selected_k == 10
    assert configured.latent_usage.collapse.delta == 0.05
    assert configured.latent_usage.collapse.epsilon_values == (0.01, 0.1)
    assert configured.topology.max_samples == 100
    assert defaults == build_metric_config({"output_dir": tmp_path, "metrics": {}})
    assert defaults.geometry.selected_k == 20
    assert defaults.latent_usage.collapse.delta == 0.01


@pytest.mark.parametrize("family, overrides", [
    ("dimensionality", {"mle_n_neighbors": 1}),
    ("geometry", {"selected_k": 0}),
    ("latent_usage", {"collapse": {"delta": 1}}),
])
def test_json_metric_overrides_use_family_validation(tmp_path, family, overrides) -> None:
    with pytest.raises(ValueError):
        build_metric_config({"output_dir": tmp_path, "metrics": {family: overrides}})


def test_central_experiment_json_builds_expected_grid() -> None:
    config = load_experiment_config(CONFIG_PATH)
    runs = build_training_runs(config["experiment_name"], config["training"])
    assert isinstance(config["experiment_name"], str)
    assert config["experiment_name"]
    assert config["output_dir"].is_absolute()
    training = config["training"]
    expected_count = math.prod(
        len(training[key])
        for key in (
            "betas", "learning_rates", "latent_dims", "seeds",
            "losses", "loss_reductions",
        )
    )
    assert len(runs) == expected_count


@pytest.mark.parametrize(
    ("dataset_id", "config_case"),
    EXPERIMENT_CONFIGS.items(),
)
def test_each_dataset_has_an_independent_central_configuration(
    dataset_id: str,
    config_case: Path,
) -> None:
    config = load_experiment_config(config_case)
    runs = build_training_runs(config["experiment_name"], config["training"])
    assert config["dataset"]["id"] == dataset_id
    assert runs
    assert {run["latent_dim"] for run in runs} == {
        int(value) for value in config["training"]["latent_dims"]
    }
    assert isinstance(config["dataset"]["options"], dict)


def test_duplicate_grid_configuration_is_removed() -> None:
    training = {
        "betas": [0.1, 0.1],
        "learning_rates": [0.001],
        "latent_dims": [2],
        "seeds": [3],
        "losses": ["kl"],
        "loss_reductions": ["mean"],
        "epochs": 1,
    }

    runs = build_training_runs("experiment", training)

    assert len(runs) == 1
    assert runs[0]["beta"] == 0.1


def test_beta_over_d_keeps_total_rate_coefficient_fixed() -> None:
    settings = load_experiment_config(CONFIG_ROOT / "experiments.json#training.tcga_beta_over_d")
    training = settings["training"]
    runs = build_training_runs(settings["experiment_name"], training)
    assert len(runs) == 400
    for dim in training["latent_dims"]:
        coefficients = sorted({r["beta"] / dim for r in runs if r["latent_dim"] == dim})
        assert coefficients == pytest.approx(sorted(training["beta_over_d"]))
    original = load_experiment_config(CONFIG_ROOT / "experiments.json#training.tcga_no_annealing")
    assert sorted({r["beta"] for r in runs if r["latent_dim"] == 8}) == pytest.approx(
        original["training"]["betas"]
    )


@pytest.mark.parametrize("change", [{"betas": [0.1]}, {"losses": ["mmd"]}, {"loss_reductions": ["sum"]}])
def test_beta_over_d_rejects_ambiguous_or_incompatible_settings(change) -> None:
    settings = load_experiment_config(CONFIG_ROOT / "experiments.json#training.tcga_beta_over_d")
    with pytest.raises(ValueError):
        build_training_runs("invalid", {**settings["training"], **change})


def test_overview_rejects_duplicate_run_names(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"run_name": ["same", "same"], "metric": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="unique non-null run_name"):
        _save_overview(frame, tmp_path)
