"""Read AUTOENCODIX's stored trainer reductions, without reconstructing a loss."""
import numpy as np
import pandas as pd


def run_native_loss_metrics_pipeline(inputs, varix):
    records = []
    epoch = inputs.metadata.get('epochs')
    result = getattr(varix, 'result', None)
    registry = getattr(result, 'sub_losses', None)
    for split in ('train', 'valid'):
        for component in ('loss', 'recon_loss', 'var_loss'):
            dynamics = getattr(result, 'losses', None) if component == 'loss' else (
                registry.get(component) if registry is not None and component in registry.keys() else None)
            # AUTOENCODIX stores zero-based epochs and prints epoch + 1.
            raw = dynamics.get(epoch=int(epoch)-1, split=split) if dynamics is not None and epoch is not None else None
            array = np.asarray(raw) if raw is not None and not isinstance(raw, dict) else np.array([])
            value = float(array.item()) if array.size == 1 else np.nan
            valid = np.isfinite(value)
            records.append(dict(metric_name=f'autoencodix_{split}_{component}',
                                value=value, status='ok' if valid else 'unavailable',
                                warning=None if valid else 'Native loss at configured final epoch unavailable; no MSE fallback.',
                                n_samples=len(inputs.x_train if split == 'train' else inputs.x_val), split=split))
    return pd.DataFrame(records)
