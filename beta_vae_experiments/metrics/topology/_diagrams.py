"""Shared helper for persistence diagrams, imported by ``__init__`` and ``landscape``.

Kept in its own module so that ``landscape`` does not import from the
partially initialised package ``__init__`` (circular import).
"""

from __future__ import annotations

import numpy as np


def _finite_diagram(diagram: np.ndarray) -> np.ndarray:
    values = np.asarray(diagram, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    values = values.reshape(-1, 2)
    return values[np.isfinite(values).all(axis=1)]
