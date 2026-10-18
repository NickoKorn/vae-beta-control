"""Paired ZADU/graph sensitivity pilot; no training or parameter selection by AUC.

Run: python -m beta_vae_experiments.result_analysis.validate_neighborhood_sensitivity

k sets the neighborhood scale; compare several k on exactly the same rows.
ZADU sources/API: https://github.com/hj-n/zadu (Jeon et al., 2023).
Leiden resolution controls community scale, not representation quality:
https://www.nature.com/articles/s41598-019-41695-z
The grids below are declared engineering choices, not published optima.
Independent dataset seeds assess sampling variation; changing Leiden seeds
on a fixed dataset assesses algorithmic variation. Do not pool these axes.
This pilot uses a noisy circle and an anisotropic deformation. It does not
establish robustness on TCGA, across beta, or predict downstream AUC.
"""
import argparse
import json
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from beta_vae_experiments.metrics.geometry import compute_zadu_geometry
from beta_vae_experiments.metrics.graph.graph_metrics_pipeline import (
    GraphMetricConfig, run_graph_metrics_pipeline,
)


def run_pilot(output_dir, n=250, seeds=(13, 17, 23), k_values=(5, 10, 20, 40),
              resolutions=(0.5, 1.0, 2.0), algorithm_seeds=(31, 47)):
    """Export all existing ZADU screening outputs and all default graph families.

    Primary geometry outputs are trustworthiness, continuity and LCMC.
    Global ZADU outputs repeated across k are controls, not replications.
    SNC uses the default seed 31 and internally floor(sqrt(n)), ignoring nominal k;
    algorithm_seeds here vary graph routines, including Leiden. Kernel widths remain unchanged.
    """
    if type(n) is not int or n < 3:
        raise ValueError("n must be an integer >= 3")
    if not k_values or len(set(k_values)) != len(k_values) or any(
            type(k) is not int or not 1 <= k <= (n-1)//2 for k in k_values):
        raise ValueError("distinct k must satisfy 1 <= k <= (n-1)//2")
    for grid in (seeds, algorithm_seeds):
        if not grid or len(set(grid)) != len(grid) or any(type(s) is not int or s < 0 for s in grid):
            raise ValueError("seeds must be distinct nonnegative integers")
    if not resolutions or len(set(resolutions)) != len(resolutions) or any(
            not np.isfinite(r) or r <= 0 for r in resolutions):
        raise ValueError("resolutions must be distinct finite positive numbers")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        angle = rng.uniform(0, 2*np.pi, n)
        x = np.column_stack((np.cos(angle), np.sin(angle))) + rng.normal(0, .01, (n, 2))
        mu = x * [1., .5]
        ids = np.array([f"seed{seed}_point{i}" for i in range(n)])
        for k in k_values:
            result = compute_zadu_geometry(x, mu, k, sample_ids=ids)
            frame = pd.DataFrame([dict(metric_name=name, value=value,
                status="ok" if np.isfinite(value) else "undefined",
                warning="; ".join(result.warnings) or None)
                for name, value in result.global_metrics.items()])
            frames.append(frame.assign(family="zadu", dataset_seed=seed, k=k,
                resolution=np.nan, algorithm_seed=31, n_samples=n))
        for resolution in resolutions:
            for algorithm_seed in algorithm_seeds:
                name = f"seed{seed}_resolution{resolution}_algorithm{algorithm_seed}"
                inputs = SimpleNamespace(run_name=name, x_test=x, mu_test=mu,
                    sample_ids_test=ids, sample_metadata_test=None, metadata={"dataset_seed": seed})
                geometry = SimpleNamespace(selected_k=k_values[0], random_state=algorithm_seed,
                    distance_metric="euclidean")
                frame = run_graph_metrics_pipeline(inputs, np.arange(n), x, mu,
                    GraphMetricConfig(k_values=tuple(k_values), leiden_resolution=resolution,
                        leiden_iterations=-1), geometry, output_dir)
                frames.append(frame.assign(family="graph", dataset_seed=seed,
                    resolution=resolution, algorithm_seed=algorithm_seed))
        print(f"Completed dataset seed {seed}, n={n}", flush=True)
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(output_dir / "sensitivity_metrics.csv", index=False)
    # Descriptive dispersion at fixed parameters; three seeds are no proof of
    # convergence. Never pool k/resolution effects with sampling variance.
    metrics.groupby(["family", "metric_name", "k", "resolution", "algorithm_seed"],
        dropna=False).value.agg(["count", "mean", "std", "min", "max"]).reset_index().to_csv(
            output_dir / "dataset_seed_dispersion.csv", index=False)
    metrics.loc[metrics.family.eq("graph")].groupby(
        ["metric_name", "dataset_seed", "resolution"], dropna=False
        ).value.agg(["count", "mean", "std", "min", "max"]).reset_index().to_csv(
            output_dir / "graph_algorithm_dispersion.csv", index=False)
    (output_dir / "sensitivity_config.json").write_text(json.dumps(dict(
        n=n, dataset_seeds=seeds, k_values=k_values, leiden_resolutions=resolutions,
        leiden_algorithm_seeds=algorithm_seeds, leiden_iterations=-1,
        zadu_snc_seed=31, zadu_snc_effective_k=int(np.sqrt(n)), noise_std=.01, deformation=[1., .5],
        versions={p: version(p) for p in ("zadu", "numpy", "igraph", "scikit-hubness")},
        scope="synthetic technical sensitivity; no TCGA/AUC validation"), indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
        default=Path("beta_vae_experiments/outputs/neighborhood_sensitivity"))
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 17, 23])
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20, 40])
    args = parser.parse_args()
    run_pilot(args.output_dir, args.n, tuple(args.seeds), tuple(args.k_values))


if __name__ == "__main__":
    main()
