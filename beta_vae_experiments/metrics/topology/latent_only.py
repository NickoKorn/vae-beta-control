"""Latent-only: deskriptive Topologie einer einzelnen Repräsentation.

Für die Masterarbeit:
Die persistente Homologie beschreibt Zusammenhangskomponenten (H0) und
Schleifen (H1) über mehrere Abstandsskalen. Balkenanzahl, Lebensdauern,
persistente Entropie und Betti-Kurven beschreiben die resultierenden Diagramme.
Diese Größen benötigen keinen Vergleich mit dem Eingaberaum und messen daher
für sich genommen keine Preservation. Eine Schleife ist kein Nachweis eines
biologischen Zyklus; eine Komponente entspricht nicht automatisch einer Klasse.

Die Hilfsfunktionen können zur Referenzdiagnostik auch X beschreiben und geben
historisch beide Räume getrennt aus. „Latent-only“ bezeichnet die Messart,
nicht das Verbot, dieselben Deskriptoren auch auf X anzuwenden.
Landscapes bleiben als optionale Funktionsdarstellungen in landscape.py.
Quellen: Bubenik (2015), https://jmlr.org/papers/v16/bubenik15a.html ;
Atienza et al., https://arxiv.org/abs/1803.08304 .

Einstieg und Quellenhierarchie:
    Munch (2017), Daten mit Distanz, PH und Mapper:
    https://doi.org/10.18608/jla.2017.42.6
    Chazal & Michel (2021), Grundlagen, Statistik und praktische TDA:
    https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2021.667963/full
    Huynh & Cang (2024), scGeom, biologische Strukturanalyse:
    https://doi.org/10.1093/bib/bbae176
    scGeom ist kein Beleg fuer X->mu-Preservation oder beta-Fruehwarnung.
    Einordnung und Methodenquellen: TDA_METHODEN_PRESERVATION_LATENT_ONLY.md.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from gudhi.representations import BettiCurve, Entropy
from ._diagrams import _finite_diagram

def summarize_persistence_diagrams(
    diagrams: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Summarize raw diagrams without turning descriptors into core predictors.

    Finite summaries use strictly positive lifetimes. Essential bars are retained
    in the raw ``.npy`` artifacts and counted here, but excluded from all finite
    diagram distances and lifetime summaries.
    """
    rows: list[dict[str, object]] = []
    for dimension in (0, 1):
        for space, key in (("x", "input"), ("mu", "latent")):
            raw = np.asarray(
                diagrams[f"topology_{key}_h{dimension}_persistence_diagram"],
                dtype=np.float64,
            ).reshape(-1, 2)
            finite = raw[np.isfinite(raw).all(axis=1)]
            finite = finite[finite[:, 1] > finite[:, 0]]
            lifetimes = finite[:, 1] - finite[:, 0]
            # Full Rips H0 normally contains one essential component. It is
            # scientific provenance, not a finite point usable by W/B matching.
            essential = np.isfinite(raw[:, 0]) & np.isposinf(raw[:, 1])
            rows.append({
                "space": space,
                "homology_dimension": dimension,
                "n_finite_bars": int(len(lifetimes)),
                "n_essential_bars": int(np.sum(essential)),
                "total_persistence": float(np.sum(lifetimes)),
                "median_lifetime": float(np.median(lifetimes)) if len(lifetimes) else np.nan,
                "max_lifetime": float(np.max(lifetimes)) if len(lifetimes) else np.nan,
                "q90_lifetime": float(np.quantile(lifetimes, 0.9)) if len(lifetimes) else np.nan,
                "empty_diagram": bool(len(lifetimes) == 0 and not np.any(essential)),
            })
    return pd.DataFrame.from_records(rows)






def compute_persistence_entropy(diagrams, n_samples):
    """GUDHI scalar Shannon entropy of positive finite bar lengths (natural log).

    Atienza et al., https://arxiv.org/abs/1803.08304. Empty diagrams are
    undefined rather than equated to a one-bar diagram (entropy zero).
    No extra division by log(feature count), no cross-space difference.

    Explorative Beschreibung eines Diagramms, kein Preservation-Abstand.
    GUDHI normalisiert die Lebensdauern für Shannon-Entropy zu Anteilen;
    normalized=False bedeutet NICHT Verwendung roher Lebensdauern als
    Wahrscheinlichkeiten. Der Parameter betrifft die Entropy-Vektorvariante.
    https://gudhi.inria.fr/python/3.11.0/representations.html
    Der Runner speichert dies separat in exploratory_entropy.csv.
    """
    records = []
    for dimension in (0, 1):
        for space, key in [('x', 'input'), ('mu', 'latent')]:
            name = f'topology_h{dimension}_persistence_entropy_{space}'
            diagram = _finite_diagram(diagrams[f'topology_{key}_h{dimension}_persistence_diagram'])
            diagram = diagram[diagram[:, 1] > diagram[:, 0]]
            if not len(diagram):
                records.append(dict(metric_name=name, value=float("nan"),
                    status="degenerate_input", warning="Persistence entropy requires positive finite lifetimes.",
                    n_samples=n_samples, split="test"))
            else:
                value = float(Entropy(mode='scalar', normalized=False).fit_transform([diagram])[0, 0])
                records.append(dict(metric_name=name, value=value, status='ok',
                                    warning=None, n_samples=n_samples, split='test'))
    return pd.DataFrame(records)



def compute_betti_curve_artifacts(
    diagrams: dict[str, np.ndarray],
) -> dict[str, np.ndarray | str]:
    """Derive shared-grid H0/H1 Betti curves from cached Ripser diagrams.

    Input representation:
        The same cached, normalized Ripser H0/H1 diagrams consumed by the
        Wasserstein comparison; essential infinite H0 bars are excluded.

    resolution=None/predefined_grid=None erzeugt laut GUDHI ein gemeinsames
    ereignisbasiertes Raster, das alle Betti-Änderungen der beiden Diagramme
    erfasst. Keine willkürliche gleichmäßige Rasterauflösung nötig:
    https://gudhi.inria.fr/python/3.11.0/representations.html
    Wegen ausgeschlossener essentieller Klasse ist H0 hier reduziert (auf
    endlichen Skalen fehlt die eine verbleibende Komponente). Gilt nicht als
    direktes Maß der Zahl biologischer Klassen. Artefaktnamen sind historisch
    noch nicht entsprechend präzisiert; der Docstring dokumentiert die Konvention.
    """
    artifacts: dict[str, np.ndarray | str] = {
        "topology_betti_curve_backend": "gudhi.representations.BettiCurve",
    }
    for dimension in (0, 1):
        input_diagram = _finite_diagram(
            diagrams[f"topology_input_h{dimension}_persistence_diagram"]
        )
        latent_diagram = _finite_diagram(
            diagrams[f"topology_latent_h{dimension}_persistence_diagram"]
        )
        if len(input_diagram) == len(latent_diagram) == 0:
            grid = np.empty(0, dtype=np.float64)
            curves = np.empty((2, 0), dtype=np.int64)
        else:
            transformer = BettiCurve(resolution=None, predefined_grid=None)
            curves = transformer.fit_transform([input_diagram, latent_diagram])
            grid = np.asarray(transformer.grid_, dtype=np.float64)
        artifacts[f"topology_betti_h{dimension}_grid"] = grid
        artifacts[f"topology_betti_h{dimension}_x"] = np.asarray(curves[0])
        artifacts[f"topology_betti_h{dimension}_mu"] = np.asarray(curves[1])
    return artifacts
