"""Reproducible technical sampling/k pilot without training or patient data.

Run from repository root:
    python -m beta_vae_experiments.result_analysis.validate_structure_metrics

Checks circle→ellipse on nested random subsets. This tests sampling/cost,
not VAE convergence or an AUC hypothesis. Chazal et al. (2014) motivates
studying estimated diagrams: https://proceedings.mlr.press/v32/chazal14.html
The chosen n/k/seeds are our pilot grid, not a published optimal setting.
ZADU is deliberately unchanged and not modified by this pilot.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from beta_vae_experiments.metrics.topology import TopologyMetricConfig, run_topology_metrics_pipeline
from beta_vae_experiments.metrics.graph.graph_metrics_pipeline import GraphMetricConfig, run_graph_metrics_pipeline


def run_pilot(output_dir, sizes=(250, 500, 1000), seeds=(13, 17, 23), k_values=(5, 10, 20)):
    """Use exact same X/mu/IDs per family, nested subsets within each seed."""
    sizes, seeds, k_values = tuple(sizes), tuple(seeds), tuple(k_values)
    if (not sizes or len(set(sizes)) != len(sizes)
            or any(type(n) is not int or n < 2 for n in sizes)):
        raise ValueError("sizes must be distinct integers >= 2")
    if not seeds or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("seeds must be nonnegative integers")
    if not k_values or any(type(k) is not int or not 1 <= k < min(sizes) for k in k_values):
        raise ValueError("Each k must be positive and smaller than all sample sizes")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames, costs = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        angle = rng.uniform(0, 2*np.pi, max(sizes))
        x = np.column_stack([np.cos(angle), np.sin(angle)])
        x += rng.normal(0, .01, x.shape)
        mu = x * np.array([1., .5])
        ids = np.array([f"seed{seed}_point{i}" for i in range(len(x))])
        order = rng.permutation(len(x))
        for n in sizes:
            indices = np.sort(order[:n])
            name = f"circle_ellipse_n{n}_seed{seed}"
            inputs = SimpleNamespace(run_name=name, x_test=x, mu_test=mu,
                                     sample_ids_test=ids, metadata={"synthetic_seed": seed})
            for family in ("topology", "graph"):
                start = time.perf_counter()
                if family == "topology":
                    frame = run_topology_metrics_pipeline(inputs, indices, x[indices], mu[indices],
                        TopologyMetricConfig(max_samples=max(sizes)), output_dir)
                else:
                    geometry = SimpleNamespace(selected_k=k_values[0], random_state=31, distance_metric="euclidean")
                    frame = run_graph_metrics_pipeline(inputs, indices, x[indices], mu[indices],
                        GraphMetricConfig(k_values=k_values), geometry, output_dir)
                elapsed = time.perf_counter()-start
                frame = frame.assign(pilot_family=family, subset_size=n, dataset_seed=seed)
                frames.append(frame)
                costs.append(dict(family=family, n=n, seed=seed, seconds=elapsed,
                                  all_ok=bool(frame.status.eq("ok").all())))
                print(f"{name}: {family}, {elapsed:.2f}s, ok={costs[-1]['all_ok']}", flush=True)
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(output_dir / "sampling_metrics.csv", index=False)
    pd.DataFrame(costs).to_csv(output_dir / "sampling_costs.csv", index=False)
    (output_dir / "pilot_config.json").write_text(json.dumps(dict(
        sizes=sizes, seeds=seeds, k_values=k_values, noise_std=.01,
        deformation=[1., .5], subsets="nested without replacement per seed",
        versions={p: version(p) for p in ("numpy", "scipy", "ripser", "persim", "gudhi", "igraph", "scikit-hubness")},
        scope="technical synthetic pilot, not evidence for TCGA/AUC or a sufficient sample size"), indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("beta_vae_experiments/outputs/structure_validation"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[250, 500, 1000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 17, 23])
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20])
    args = parser.parse_args()
    run_pilot(args.output_dir, args.sizes, args.seeds, args.k_values)


if __name__ == "__main__":
    main()
