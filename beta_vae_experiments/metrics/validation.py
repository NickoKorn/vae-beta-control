"""Strict validation for aligned absolute-core metric inputs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .contracts import CoreMetricInputs


def _matrix(
    value: np.ndarray | None, name: str, *, optional: bool = False
) -> np.ndarray | None:
    if value is None and optional:
        return None
    if value is None:
        raise ValueError(f"{name} is required")
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must have a numeric dtype")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _sample_ids(
    values: np.ndarray | None,
    expected_rows: int,
    split: str,
) -> np.ndarray | None:
    if values is None:
        return None
    ids = np.asarray(values, dtype=object)
    if ids.ndim != 1 or len(ids) != expected_rows:
        raise ValueError(
            f"sample_ids_{split} must be one-dimensional with {expected_rows} rows"
        )
    normalized = np.asarray([str(value) for value in ids], dtype=object)
    if any(not value.strip() for value in normalized):
        raise ValueError(f"sample_ids_{split} contains empty IDs")
    if len(set(normalized.tolist())) != len(normalized):
        raise ValueError(f"sample_ids_{split} contains duplicate IDs")
    return normalized


def validate_metric_inputs(inputs: CoreMetricInputs) -> CoreMetricInputs:
    """Validate the normalized contract without silently reordering observations."""
    if not str(inputs.run_name).strip():
        raise ValueError("run_name must be non-empty")

    values: dict[str, np.ndarray | None] = {}
    for split in ("train", "val", "test"):
        x = _matrix(getattr(inputs, f"x_{split}"), f"x_{split}")
        recon = _matrix(getattr(inputs, f"x_recon_{split}"), f"x_recon_{split}")
        mu = _matrix(getattr(inputs, f"mu_{split}"), f"mu_{split}")
        if x.shape != recon.shape:
            raise ValueError(
                f"x_{split} and x_recon_{split} shapes differ: {x.shape} versus {recon.shape}"
            )
        if x.shape[0] != mu.shape[0]:
            raise ValueError(
                f"x_{split} and mu_{split} row counts differ: {x.shape[0]} versus {mu.shape[0]}"
            )
        values[f"x_{split}"] = x
        values[f"x_recon_{split}"] = recon
        values[f"mu_{split}"] = mu

        for optional_name in ("logvar", "z"):
            optional = _matrix(
                getattr(inputs, f"{optional_name}_{split}"),
                f"{optional_name}_{split}",
                optional=True,
            )
            if optional is not None and optional.shape != mu.shape:
                raise ValueError(
                    f"{optional_name}_{split} must match mu_{split} shape "
                    f"{mu.shape}, got {optional.shape}"
                )
            values[f"{optional_name}_{split}"] = optional
        values[f"sample_ids_{split}"] = _sample_ids(
            getattr(inputs, f"sample_ids_{split}"),
            x.shape[0],
            split,
        )

    input_dimensions = {
        values[f"x_{split}"].shape[1] for split in ("train", "val", "test")
    }
    latent_dimensions = {
        values[f"mu_{split}"].shape[1] for split in ("train", "val", "test")
    }
    if len(input_dimensions) != 1:
        raise ValueError("input feature dimensions differ across splits")
    if len(latent_dimensions) != 1:
        raise ValueError("latent dimensions differ across splits")

    return replace(inputs, **values)
