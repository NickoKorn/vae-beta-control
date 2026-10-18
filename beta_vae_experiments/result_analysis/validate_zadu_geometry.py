"""Technical ZADU audit without training; original screening metrics retained.

Run: python -m beta_vae_experiments.result_analysis.validate_zadu_geometry

Declared n/k/seed grid is a pilot design, not an optimum from the literature.
Sources/definitions are beside the geometry implementation and in each manifest.
Nested sample rows within seed permit paired comparisons across subset sizes.
A noisy curve in 8D tests ordinary inputs; duplicates/collapse test failure policy.
No output here establishes predictive utility for TCGA downstream AUC on z.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from beta_vae_experiments.metrics.geometry import GeometryMetricConfig, run_geometry_metrics_pipeline


def run_pilot(output_dir, sizes=(128, 250, 500), seeds=(13, 17), ks=(5, 10, 20, 40)):
    if (not sizes or len(set(sizes)) != len(sizes)
            or any(type(n) is not int or n < 3 for n in sizes)):
        raise ValueError('sizes must be distinct integers >=3')
    if not ks or len(set(ks)) != len(ks) or any(type(k) is not int or not 1 <= k <= (min(sizes)-1)//2 for k in ks):
        raise ValueError('k grid must be distinct valid neighborhood sizes')
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('seeds must be distinct nonnegative integers')
    output_dir = Path(output_dir)
    frames = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        angle = rng.uniform(0, 2*np.pi, max(sizes))
        x = np.column_stack([np.cos(angle), np.sin(angle)])
        x += rng.normal(0, .01, x.shape)
        x = np.column_stack([x, rng.normal(0, .01, (len(x), 6))])
        ids = np.array([f'seed{seed}_p{i}' for i in range(len(x))])
        order = rng.permutation(len(x))
        for n in sizes:
            indices = np.sort(order[:n])
            selected = x[indices]
            for case in ('identity', 'scale', 'deform', 'permuted', 'duplicates', 'collapse'):
                mu = selected.copy()
                if case == 'scale': mu *= 1e-8
                if case == 'deform': mu[:, 1] *= .3
                if case == 'permuted': mu = mu[rng.permutation(n)]
                if case == 'duplicates': mu[1] = mu[0]
                if case == 'collapse': mu[:] = 0
                full_mu = x.copy()
                full_mu[indices] = mu
                for k in ks:
                    name = f'{case}_n{n}_seed{seed}_k{k}'
                    inputs = SimpleNamespace(run_name=name, x_test=x, mu_test=full_mu,
                        sample_ids_test=ids)
                    frame = run_geometry_metrics_pipeline(inputs, indices, selected, mu,
                        GeometryMetricConfig(selected_k=k, random_state=31), output_dir)
                    frames.append(frame.assign(case=case, n=n, dataset_seed=seed, k=k))
            print(f'Completed n={n}, dataset_seed={seed}', flush=True)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(output_dir/'zadu_validation_metrics.csv', index=False)
    result.groupby(['case', 'n', 'k', 'status']).size().rename('count').reset_index().to_csv(
        output_dir/'zadu_validation_statuses.csv', index=False)
    (output_dir/'validation_config.json').write_text(json.dumps(dict(
        sizes=sizes, dataset_seeds=seeds, k_values=ks, snc_seed=31,
        scope='technical synthetic validation, not TCGA/AUC validation'), indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
        default=Path('beta_vae_experiments/outputs/zadu_validation'))
    args = parser.parse_args()
    run_pilot(args.output_dir)
