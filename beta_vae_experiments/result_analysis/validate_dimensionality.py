"""Reproduzierbarer synthetischer n-/Noise-/Parameter-Pilot, kein AUC-Nachweis.

Aufruf: python -m beta_vae_experiments.result_analysis.validate_dimensionality
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from beta_vae_experiments.metrics.dimensionality import (
    DimensionalityMetricConfig, _estimate_intrinsic_dimension, compute_effective_rank_mu,
)


def run(output_dir):
    rows = []
    for seed in (13, 17):
        for n in (250, 500, 1000):
            rng = np.random.default_rng(np.random.SeedSequence([seed, n]))
            rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
            angle = rng.uniform(0, 2*np.pi, n)
            plane = rng.uniform(-1, 1, (n, 2))
            cases = {'line': (np.column_stack([plane[:, 0], np.zeros(n)]), 1),
                     'circle': (np.column_stack([np.cos(angle), np.sin(angle)]), 1),
                     'plane': (plane, 2)}
            for case, (coordinates, intrinsic_dim) in cases.items():
                clean = coordinates @ rotation[:2]
                noise = rng.standard_normal(clean.shape)
                for noise_sd in (0., .01, .1):
                    points = clean + noise_sd * noise
                    common = dict(case=case, seed=seed, n_samples=n, noise_sd=noise_sd,
                                  noiseless_intrinsic_dimension=intrinsic_dim)
                    for metric in compute_effective_rank_mu(points):
                        rows.append(dict(common, parameter=None, **metric))
                    for discard in (0., .1, .2):
                        result = _estimate_intrinsic_dimension(points, representation='mu', estimator_name='twonn',
                            config=DimensionalityMetricConfig(twonn_discard_fraction=discard))
                        rows.append(dict(common, parameter=discard, **result))
                    for k in (10, 20, 40):
                        result = _estimate_intrinsic_dimension(points, representation='mu', estimator_name='mle',
                            config=DimensionalityMetricConfig(mle_n_neighbors=k))
                        rows.append(dict(common, parameter=k, **result))
    frame = pd.DataFrame(rows)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / 'metrics.csv', index=False)
    summary = dict(n_rows=len(frame), n_unavailable=int((frame.status != 'ok').sum()),
        purpose='Synthetic sensitivity only; no TCGA or AUC validity claim',
        ground_truth='Noiseless manifold dimension only; nonzero ambient noise changes local dimension',
        seeds=[13, 17], n_samples=[250, 500, 1000], noise_sd=[0., .01, .1],
        twonn_discard_fraction=[0., .1, .2], mle_n_neighbors=[10, 20, 40])
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parents[1] / 'outputs' / 'dimensionality_validation')
    run(parser.parse_args().output_dir)
