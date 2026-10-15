import numpy as np
import pytest
import pandas as pd

from beta_vae_experiments.result_analysis.sweep_analysis import (
    collapse_boundaries,
    local_rd_slopes,
    pareto_mask,
    save_sweep_analysis,
)
from beta_vae_experiments.metrics.latent_usage import PosteriorCollapseConfig
from beta_vae_experiments.metrics.latent_usage import compute_epsilon_delta_posterior_collapse


def test_rate_distortion_pareto_and_slope_guard() -> None:
    rate = np.array([1.0, 2.0, 3.0, 4.0])
    distortion = np.array([4.0, 3.0, 3.5, 1.0])
    assert pareto_mask(rate, distortion).tolist() == [True, True, False, True]
    slopes = local_rd_slopes(np.array([1.0, 1.0, 2.0]), np.array([3.0, 2.0, 1.0]))
    assert np.isnan(slopes[0])
    assert slopes[1] == pytest.approx(1.0)


def test_epsilon_delta_collapse_uses_strict_kl_and_inclusive_fraction_threshold() -> None:
    # delta=.25 requires at least 3/4 samples strictly below epsilon.
    kl = np.array([
        [0.01, 0.009],
        [0.009, 0.009],
        [0.009, 0.009],
        [0.009, 0.02],
    ])
    config = PosteriorCollapseConfig(
        delta=0.25, epsilon_values=(0.01,), primary_epsilon=0.01
    )
    result = compute_epsilon_delta_posterior_collapse(kl, "toy", config)
    # First dimension has exactly 3/4 below epsilon; equality at epsilon is excluded.
    indexed = result.set_index("metric_name")["value"]
    assert next(value for name, value in indexed.items() if "_count_" in name) == 2
    assert next(value for name, value in indexed.items() if "_fraction_" in name) == 1.0
    assert result["n_samples"].eq(4).all()
    assert result["split"].eq("test").all()


def test_collapse_boundaries_persistent_and_censored_with_missing_value() -> None:
    frame = pd.DataFrame({
        "dataset_id": "toy", "learning_rate": 5e-4, "latent_dim": 4, "seed": 2,
        "beta": [0.0, 0.001, 0.002, 0.004, 0.008],
        "posterior_collapsed_dimensions_fraction": [0.0, 0.2, np.nan, 0.6, 0.7],
    })
    row = collapse_boundaries(frame).loc[lambda x: x["seed"].eq(2)].iloc[0]
    assert row["beta_collapse_0p1"] == pytest.approx(0.001)
    assert row["beta_collapse_0p1_persistent"] == pytest.approx(0.004)
    assert row["beta_collapse_0p5"] == pytest.approx(0.004)
    assert row["beta_collapse_0p5_persistent"] == pytest.approx(0.004)


def test_sweep_level_analysis_writes_required_outputs(tmp_path) -> None:
    rows = []
    metrics = (
        "info_mutual_information_nats", "posterior_mpd_mean_nats",
        "posterior_snr_mean", "dimensionality_mu_effective_rank_fraction",
        "geometry_mu_lcmc_zadu_k10", "topology_nts_m", "topology_srtd_lite",
        "posterior_collapsed_dimensions_fraction",
    )
    for seed in (2, 10, 21):
        for index, beta in enumerate((0.0, 1e-4, 3e-4, 1e-3, 1e-2)):
            row = {
                "dataset_id": "toy", "learning_rate": 5e-4, "latent_dim": 4,
                "seed": seed, "beta": beta, "status": "ok",
                "posterior_conditional_kl_total_mean": 5-index,
                "autoencodix_valid_recon_loss": 1+index,
            }
            row.update({metric: float(index + seed / 100) for metric in metrics})
            rows.append(row)
    save_sweep_analysis(pd.DataFrame(rows), tmp_path)
    for filename in (
        "rate_distortion_summary.csv", "collapse_boundary_summary.csv",
    ):
        assert (tmp_path / filename).is_file()
