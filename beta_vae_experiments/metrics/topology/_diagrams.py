"""Shared helper for persistence diagrams, imported by ``__init__`` and ``landscape``.

Kept in its own module so that ``landscape`` does not import from the
partially initialised package ``__init__`` (circular import).
"""

from __future__ import annotations

import numpy as np


def _finite_diagram(diagram: np.ndarray) -> np.ndarray:
    """Endliche Klassen für Diagrammabstände und Vektorartefakte auswählen.

    Bei vollständiger Rips-Filtration bleibt für eine nichtleere endliche
    Punktwolke eine essentielle H0-Klasse (0,infinity). Sie bleibt im
    Rohdiagramm; endliche Vergleiche verwenden diese gefilterte Ansicht.
    Mathematische Grundlage: https://arxiv.org/abs/1207.3885

    NaN, nichtendliche Geburt und Tod vor Geburt werden als ungültig abgelehnt;
    diese Fehler dürfen nicht als leeres Diagramm oder essentielle Klasse gelten.
    Das leere Array hat absichtlich Shape (0,2): leeres H1 ist ein gültiger Fall.
    """
    values = np.asarray(diagram, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Persistence diagrams must have shape (n, 2)")
    if np.isnan(values).any() or not np.isfinite(values[:, 0]).all():
        raise ValueError("Invalid persistence birth or NaN coordinate")
    if np.isneginf(values[:, 1]).any() or (values[:, 1] < values[:, 0]).any():
        raise ValueError("Invalid persistence death coordinate")
    return values[np.isfinite(values).all(axis=1)]
