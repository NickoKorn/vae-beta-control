"""Focused structural comparison of aligned input data ``X`` and posterior means ``mu``.

Preservation für die Masterarbeit:
Wir vergleichen die metrische Struktur derselben Patienten im vorverarbeiteten
Eingaberaum X und in den deterministischen Posterior-Mittelwerten mu.
H0/H1-Diagrammabstände vergleichen topologische Signaturen ohne Punktidentität.
Der H0-Signature-Vergleich nach Moor ist eine lokale Complementary-Analyse
und wird von der aktiven Pipeline nicht ausgeführt.
Diese Auswertung erfolgt nach dem Training und verändert den VAE-Loss nicht.
Kleine Fehler bedeuten stärkere Erhaltung der jeweils gemessenen Struktur,
belegen aber weder biologische Richtigkeit noch erhaltene Downstream-AUC.
Separate Durchmesser-Normalisierung entfernt gleichmäßige Skalierung; die
absoluten Skalen werden deshalb zusätzlich protokolliert.
Persistence entropy, landscapes, diagrams, Betti curves, and bar summaries are
diagnostic artifacts. SRTD/NTS is archived outside the active pipeline.
This block compares X -> mu, independently of posterior-distribution topology.

Wissenschaftliche Begründung:
    Rieck & Leitte (2015) vergleichen persistente Qualitätsfunktionen
    (insbesondere Dichte), nicht direkt unseren Rips-H1-Diagrammabstand:
    https://bastian.rieck.me/research/EuroVis2015.pdf
    Chazal, de Silva & Oudot begründen Rips-Persistenz und deren Stabilität:
    https://arxiv.org/abs/1207.3885
    Carriere et al. (2021) vergleichen direkt Rips-Diagramme von Input/Latent:
    https://proceedings.mlr.press/v139/carriere21a.html
    Moor et al. (2020) verwenden einen verwandten topologischen Trainings-Loss,
    in den Hauptversuchen auf H0-Kantenpaarungen; nicht unseren H1-Wasserstein:
    https://proceedings.mlr.press/v119/moor20a/moor20a.pdf

    Unsere Frage ist die Erhaltung relativer mehrskaliger Konnektivität und
    Schleifenstruktur. Dass H1-Wasserstein Aufgabenverluste früh anzeigt, bleibt
    eine empirische Hypothese. Ein Bibliotheksdefault ist kein Optimalitätsbeleg.
    Persistenzdiagramme vergessen die Patientenkorrespondenz: Die Permutation
    einer Punktwolke ändert ihr Diagramm nicht. ZADU/Downstream ergänzen diese
    globale, nicht korrespondenzsensitive Strukturbeschreibung.

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

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import json
import hashlib
import os
import tempfile
import warnings
import numpy as np
import pandas as pd
from ._diagrams import _finite_diagram
from .latent_descriptor import (summarize_persistence_diagrams, compute_persistence_entropy,
                          compute_betti_curve_artifacts)
from .landscape import LandscapeConfig, run_landscape_metrics, NORMALIZATION
from gudhi import bottleneck_distance
from gudhi.wasserstein import wasserstein_distance
from sklearn.metrics import pairwise_distances
from typing import TYPE_CHECKING

try:
    from ripser import ripser
except ImportError:  # pragma: no cover
    ripser = None

try:
    from persim import wasserstein
except ImportError:  # pragma: no cover
    wasserstein = None

if TYPE_CHECKING:
    from beta_vae_experiments.metric_pipeline import CoreMetricInputs

WASSERSTEIN_METRIC_NAMES = (
    "topology_h0_normalized_wasserstein",
    "topology_h1_normalized_wasserstein",
)
WASSERSTEIN_W2_METRIC_NAMES = (
    "topology_h0_normalized_wasserstein_w2",
    "topology_h1_normalized_wasserstein_w2",
)
BOTTLENECK_METRIC_NAMES = (
    "topology_h0_normalized_bottleneck",
    "topology_h1_normalized_bottleneck",
)
ENTROPY_METRIC_NAMES = tuple(
    f"topology_h{dim}_persistence_entropy_{space}"
    for dim in (0, 1) for space in ("x", "mu")
)
PERSISTENCE_METRIC_NAMES = WASSERSTEIN_METRIC_NAMES + WASSERSTEIN_W2_METRIC_NAMES + BOTTLENECK_METRIC_NAMES
TOPOLOGY_METRIC_NAMES = PERSISTENCE_METRIC_NAMES

# Pin even Ripser's current defaults: silent upstream-default changes must not
# alter persistence diagrams between otherwise identical experiment runs.
# Parametersemantik für den Pin 0.6.15:
# https://ripser.scikit-tda.org/en/latest/reference/stubs/ripser.ripser.html
# Rips benötigt paarweise Distanzen, keine gleiche Ambientdimension von X/μ.
# Die euklidische Ausgangsmetrik wird VORHER mit pairwise_distances berechnet;
# Ripser berechnet bei distance_matrix=True keine neue Featuredistanz.
RIPSER_CONFIG = {
    # Sonst würden Distanzmatrixzeilen fälschlich als Featurevektoren gelten.
    "distance_matrix": True,
    # H0: Komponentenverschmelzung; H1: Schleifen über Filtrationsskalen.
    # Homologiedimension, nicht Latentdimension. H2 der Sphäre ist ausgeschlossen.
    "maxdim": 1,
    # Koeffizientenkörper F2: feste, einfache Wahl ohne Orientierungszeichen.
    # Kein Beleg allgemeiner Überlegenheit oder Koeffizientenunabhängigkeit.
    "coeff": 2,
    # Vollständige Filtration vermeidet künstlich rechtszensierte Balken.
    # Bei endlichen dichten Distanzen ist am Durchmesser der volle Simplex
    # erreicht; positive Homologie verschwindet spätestens dort.
    "thresh": np.inf,
    # Keine zusätzliche Landmark-Approximation IM ausgewählten Subset.
    # Das äußere Subsampling bleibt bestehen; keine Populations-Exaktheit.
    "n_perm": None,
    "do_cocycles": False,
}
# do_cocycles=False ist explizit: Für Diagrammabstände brauchen wir keine
# repräsentativen Cocyklen.

@dataclass(frozen=True)
class TopologyMetricConfig:
    """Controls for the focused structural and persistence topology block.

    Die Zahlen unten sind Projektentscheidungen, keine literaturbewiesenen
    Optima. Chazal et al. (2014) begründen statistische Diagrammschätzung,
    nicht einen universellen Mindestumfang von 1000 Patienten:
    https://proceedings.mlr.press/v32/chazal14.html

    Vor Large Scale: Streuung/Kosten auf 250/500/1000 Samples und mehreren
    Subsets prüfen; diese n-Stufen sind unser Pilotraster. X-vs-X-Subsets
    beschreiben Samplingvariation, sind kein automatischer Signifikanztest.
    Numerische Guards separat von fachlichem Posterior Collapse behandeln.
    """

    # Vorläufiges Rechenbudget. Tatsächliches n wird in _select_test_rows auch
    # durch geometry.max_samples und die vorhandene Splitgröße begrenzt.
    max_samples: int = 1000
    # Reproduzierbare äußere Auswahl; 31 besitzt keine fachliche Sonderrolle.
    # Metrik-Seed von Trainingsseed unterscheiden und geordnete IDs speichern.
    random_state: int = 31
    # Schutz vor Division durch (nahe) Null, KEINE Collapse-Definition.
    # 1e-12 nach tatsächlicher Distanzskala/Präzision im Pilot überprüfen;
    # ein Float64-Array garantiert keine Float64-Präzision aller Backends.
    diameter_epsilon: float = 1e-12
    # Accepted only for compatibility with historical configurations; unused.
    srtd_quantile: float = 0.90
    # Kompatibilitätsoption: steuert im aktuellen Runner keine PH-Plots.
    ph_visualizations_enabled: bool = False
    # Explorative Artefakte; Begründung und Grenzen in LandscapeConfig.
    landscape_num_landscapes: int = 5
    landscape_resolution: int = 100
    landscapes_enabled: bool = False

    def __post_init__(self) -> None:
        LandscapeConfig(self.landscape_num_landscapes, self.landscape_resolution)
        if type(self.max_samples) is not int or self.max_samples < 2:
            raise ValueError("topology max_samples must be at least two")
        if not np.isfinite(self.diameter_epsilon) or self.diameter_epsilon <= 0:
            raise ValueError("topology diameter_epsilon must be positive")
        if type(self.random_state) is not int or self.random_state < 0:
            raise ValueError("topology random_state must be a nonnegative integer")
        if type(self.landscapes_enabled) is not bool:
            raise TypeError("landscapes_enabled must be a boolean")
        if not 0 < self.srtd_quantile <= 1:
            raise ValueError("topology srtd_quantile must lie in (0, 1]")
        if type(self.ph_visualizations_enabled) is not bool:
            raise TypeError("ph_visualizations_enabled must be a boolean")



def validate_paired_dissimilarity_matrices(
    input_distances: np.ndarray,
    latent_distances: np.ndarray,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-8,
    negative_tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and canonically clean paired dissimilarity matrices.

    atol/rtol/negative_tolerance erlauben kleine Rundungsabweichungen, nicht
    materielle Asymmetrie oder negative Abstände. Ihre konkreten Werte sind
    technische Projektentscheidungen, keine PH-Stabilitätsschranken aus Papers.
    Symmetrisierung/Nullsetzen erfolgt erst nach erfolgreicher Prüfung.
    Endlichkeit und erwartete Zahlenbereiche im Pilot absichern. Diese Prüfung
    allein beweist weder Dreiecksungleichung noch korrekte Patientenreihenfolge.
    """
    matrices = (
        np.asarray(input_distances, dtype=np.float64),
        np.asarray(latent_distances, dtype=np.float64),
    )
    for name, matrix in zip(("input_distances", "latent_distances"), matrices):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} must be a square two-dimensional matrix")
        if len(matrix) < 2:
            raise ValueError(f"{name} must contain at least two observations")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} must contain only finite values")
        if float(np.min(matrix)) < -negative_tolerance:
            raise ValueError(f"{name} contains materially negative dissimilarities")
        if not np.allclose(matrix, matrix.T, atol=atol, rtol=rtol):
            raise ValueError(f"{name} is materially asymmetric")
        if not np.allclose(np.diag(matrix), 0.0, atol=atol, rtol=0.0):
            raise ValueError(f"{name} must have an approximately zero diagonal")
    if matrices[0].shape != matrices[1].shape:
        raise ValueError("Input and latent dissimilarity matrices must have equal shapes")

    canonical = []
    for matrix in matrices:
        cleaned = np.maximum(0.5 * (matrix + matrix.T), 0.0)
        np.fill_diagonal(cleaned, 0.0)
        canonical.append(cleaned)
    return canonical[0], canonical[1]


def _dependency_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unavailable"


def _undefined_values(
    status: str,
    warning: str,
    n_samples: int,
    names: tuple[str, ...] = TOPOLOGY_METRIC_NAMES,
) -> list[dict[str, object]]:
    return [
        {"metric_name": name, "value": float("nan"), "status": status, "warning": warning, "n_samples": n_samples, "split": "test"}
        for name in names
    ]


def compute_persistence_diagrams_h0_h1(
    input_distances: np.ndarray,
    latent_distances: np.ndarray,
    input_diameter: float,
    latent_diameter: float,
    input_diagrams: list[np.ndarray] | None = None,
) -> dict[str, np.ndarray] | None:
    """Compute normalized X/mu H0/H1 diagrams exactly once with Ripser.

    Input representation:
        Paired Euclidean dissimilarity matrices from aligned ``X_test`` and
        ``mu_test`` rows, normalized separately by their maximum distances.

    Warum separate Durchmesser-Normalisierung?
        Eigene direkte Ableitung: Wenn D_mu = c * D_X mit c > 0, sind die
        normalisierten Matrizen identisch und erzeugen gleiche Diagramme.
        So vergleichen wir relative metrische Form unabhängig von einheitlicher
        Skalierung. X- und μ-Abstände besitzen keine automatisch gleichen
        Einheiten. Die Rips-Theorie begründet die Filtration, aber nicht den
        Durchmesser als universell optimale Normierung.

    Grenzen und Nachweise:
        Kontraktion wird absichtlich entfernt. Ausreißer bestimmen den Nenner;
        fast kollabierte Räume können kleine Restunterschiede stark vergrößern.
        Deshalb absolute Skalen getrennt speichern und Identität, Skalierung,
        Ausreißer sowie Null-/Nahe-Null-Durchmesser prüfen. Der Runner prüft den
        Durchmesser vor diesem Aufruf; diese Hilfsfunktion prüft ihn nicht selbst.
        Vollständige Diagramme behalten die essentielle H0-Klasse (0, infinity).
    """
    if ripser is None:
        warning = "ripser is required for persistent-homology artifacts."
        print(warning)
        return None
    if not all(np.isfinite(d) and d > 0 for d in (input_diameter, latent_diameter)):
        raise ValueError("Persistence normalization requires positive finite diameters")
    if input_diagrams is None:
        input_diagrams = ripser(input_distances / input_diameter, **RIPSER_CONFIG)["dgms"]
    latent_diagrams = ripser(latent_distances / latent_diameter, **RIPSER_CONFIG)["dgms"]
    return {
            "topology_input_h0_persistence_diagram": np.asarray(
                input_diagrams[0], dtype=np.float64
            ),
            "topology_input_h1_persistence_diagram": np.asarray(
                input_diagrams[1], dtype=np.float64
            ),
            "topology_latent_h0_persistence_diagram": np.asarray(
                latent_diagrams[0], dtype=np.float64
            ),
            "topology_latent_h1_persistence_diagram": np.asarray(
                latent_diagrams[1], dtype=np.float64
            ),
        }

def compute_normalized_wasserstein_h0_h1(
    diagrams: dict[str, np.ndarray],
    n_samples: int,
) -> pd.DataFrame:
    """Compare cached normalized H0/H1 diagrams with Wasserstein distance.

    Input representation:
        Ripser persistence diagrams already computed from the separately
        diameter-normalized paired ``X_test`` and ``mu_test`` metric spaces.

    Definition und Begründung:
        Persim verwendet ein minimales Summenmatching (W1) mit euklidischen
        Punktkosten und orthogonalen Diagonalkosten. So werden Änderungen
        mehrerer Merkmale akkumuliert; der Score hängt auch von ihrer Zahl ab.
        matching=False betrifft nur die Rückgabe, nicht die Optimierung.
        Offizieller Referenzcode (installierten Pin 0.3.8 zusätzlich prüfen):
        https://github.com/scikit-tda/persim/blob/master/persim/wasserstein.py

    Unabhängige Kontrollfälle:
        Identische oder zwei leere Diagramme: Abstand 0. Ein Balken (b,d)
        gegen leer: (d-b)/sqrt(2). Leeres H1 ist gültig, kein Paketfehler.
        Keine allgemeine AUC- oder Wasserstein-Stabilitätsgarantie allein aus
        Bottleneck-Stabilität ableiten. Nicht mit Patientenposterior-W2 verwechseln.
    """
    if wasserstein is None:
        warning = "persim is required for topology Wasserstein metrics."
        return pd.DataFrame.from_records(_undefined_values(
                "missing_dependency", warning, n_samples,
                WASSERSTEIN_METRIC_NAMES,
            ))
    values: list[dict[str, object]] = []
    for dimension, name in enumerate(WASSERSTEIN_METRIC_NAMES):
        input_finite = _finite_diagram(
            diagrams[f"topology_input_h{dimension}_persistence_diagram"]
        )
        latent_finite = _finite_diagram(
            diagrams[f"topology_latent_h{dimension}_persistence_diagram"]
        )
        distance = (
            0.0
            if len(input_finite) == len(latent_finite) == 0
            else float(wasserstein(input_finite, latent_finite, matching=False))
        )
        values.append({"metric_name": name, "value": distance, "status": "ok", "warning": None, "n_samples": n_samples, "split": "test"})
    return pd.DataFrame.from_records(values)


def compute_normalized_wasserstein_w2_h0_h1(
    diagrams: dict[str, np.ndarray], n_samples: int,
) -> pd.DataFrame:
    """Compare finite normalized diagrams using W2 with Euclidean ground costs.

    This matches diagram points (including diagonal matches), not patients or
    Gaussian posteriors. Uses the same cached diagrams as W1 and bottleneck.
    https://gudhi.inria.fr/python/3.12.0/wasserstein_distance_user.html
    """
    rows = []
    for dimension, name in enumerate(WASSERSTEIN_W2_METRIC_NAMES):
        dx = _finite_diagram(diagrams[f"topology_input_h{dimension}_persistence_diagram"])
        dm = _finite_diagram(diagrams[f"topology_latent_h{dimension}_persistence_diagram"])
        value = float(wasserstein_distance(
            dx, dm, order=2, internal_p=2, keep_essential_parts=False,
        ))
        if not np.isfinite(value):
            raise ValueError("Non-finite diagram W2 distance")
        rows.append(dict(metric_name=name, value=value, status="ok", warning=None,
                         n_samples=n_samples, split="test"))
    return pd.DataFrame.from_records(rows)


def compute_normalized_bottleneck_h0_h1(
    diagrams: dict[str, np.ndarray], n_samples: int,
) -> pd.DataFrame:
    """GUDHI bottleneck on cached diameter-normalized finite diagram points.

    https://gudhi.inria.fr/python/3.11.0/bottleneck_distance_user.html
    """
    rows = []
    for dimension, name in enumerate(BOTTLENECK_METRIC_NAMES):
        try:
            dx = _finite_diagram(diagrams[f"topology_input_h{dimension}_persistence_diagram"])
            dm = _finite_diagram(diagrams[f"topology_latent_h{dimension}_persistence_diagram"])
            value = 0.0 if len(dx) == len(dm) == 0 else float(bottleneck_distance(dx, dm, e=0.0))
            if not np.isfinite(value):
                raise ValueError("Non-finite bottleneck distance")
            rows.append(dict(metric_name=name, value=value, status="ok", warning=None,
                             n_samples=n_samples, split="test"))
        except (ValueError, RuntimeError) as error:
            rows.extend(_undefined_values("degenerate_input", f"Bottleneck failed: {error}", n_samples, (name,)))
    return pd.DataFrame.from_records(rows)

def _cached_input_diagrams(distances, ids, directory):
    """Cache only PH(X), keyed by exact data, ordered IDs and backend settings.

    No approximation is introduced. A different subset/preprocessing changes
    the key; package versions and normalization prevent incompatible reuse.
    Atomic publication avoids partially written cache files on interruption.
    """
    if ripser is None:
        return None, None
    facts = dict(ids=None if ids is None else list(map(str, ids)),
                 shape=distances.shape, normalization=NORMALIZATION,
                 ripser_config={**RIPSER_CONFIG, "thresh": "inf"},
                 ripser_version=_dependency_version("ripser"))
    key = hashlib.sha256(json.dumps(facts, sort_keys=True).encode()
                         + np.ascontiguousarray(distances).tobytes()).hexdigest()
    path = None if directory is None else directory / (key + ".npz")
    if path is not None and path.exists():
        try:
            with np.load(path, allow_pickle=False) as archive:
                diagrams = [archive["h0"], archive["h1"]]
            for diagram in diagrams:
                _finite_diagram(diagram)
            return diagrams, key
        except (ValueError, OSError, KeyError, EOFError):
            warnings.warn("Invalid PH(X) cache; recomputing", RuntimeWarning)
    diagrams = ripser(distances / distances.max(), **RIPSER_CONFIG)["dgms"]
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
        try:
            with os.fdopen(handle, "wb") as stream:
                np.savez_compressed(stream, h0=diagrams[0], h1=diagrams[1])
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return diagrams, key


def compute_distance_scale_diagnostics(
    input_distances: np.ndarray,
    latent_distances: np.ndarray,
) -> dict[str, float | None]:
    """Record absolute scale that separate diameter normalization removes.

    Durchmesser/Median/q90 zeigen die im PH-Vergleich entfernte Skala und
    mögliche Ausreißerdominanz. Das X/μ-Verhältnis ist wegen unterschiedlicher
    Koordinatenkonventionen keine universell physikalische Kontraktionsgröße.
    Der Runner speichert diese Angaben auch vor einem Degenerationsabbruch,
    damit vollständiger Collapse nicht gerade die Skalenartefakte entfernt.
    """
    # Use each unordered pair once and omit the artificial zero diagonal.
    upper = np.triu_indices_from(input_distances, k=1)
    input_values = input_distances[upper]
    latent_values = latent_distances[upper]
    input_diameter = float(np.max(input_values))
    latent_diameter = float(np.max(latent_values))
    return {
        "input_diameter": input_diameter,
        "latent_diameter": latent_diameter,
        "latent_to_input_diameter_ratio": latent_diameter / input_diameter if input_diameter > 0 else None,
        "input_distance_median": float(np.median(input_values)),
        "input_distance_q90": float(np.quantile(input_values, 0.9)),
        "latent_distance_median": float(np.median(latent_values)),
        "latent_distance_q90": float(np.quantile(latent_values, 0.9)),
    }


def run_topology_metrics_pipeline(
    inputs: CoreMetricInputs,
    selected_indices: np.ndarray,
    x_selected: np.ndarray,
    mu_selected: np.ndarray,
    config: TopologyMetricConfig,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Compare aligned X/mean geometry; preserve scale facts even at collapse.

    Primary outputs are six diagram distances. Entropy, Betti curves and
    optional landscapes are separate diagnostics, never primary predictors.
    Separate diameter normalization measures relative shape, not contraction.
    Sources and reasons for each parameter are beside RIPSER_CONFIG above.
    """
    x = np.asarray(x_selected, dtype=np.float64)
    mu = np.asarray(mu_selected, dtype=np.float64)
    indices = np.asarray(selected_indices)
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or len(np.unique(indices)) != len(indices) or (indices < 0).any()
            or (indices >= len(inputs.x_test)).any()):
        raise ValueError("Selected indices must be unique valid test indices")
    if not np.array_equal(np.asarray(inputs.x_test)[indices], x, equal_nan=True):
        raise ValueError("Selected X does not match declared test rows")
    if not np.array_equal(np.asarray(inputs.mu_test)[indices], mu, equal_nan=True):
        raise ValueError("Selected mu does not match declared test rows")
    ids = getattr(inputs, "sample_ids_test", None)
    selected_ids = None if ids is None else np.asarray(ids)[indices].astype(str)
    if selected_ids is not None and len(np.unique(selected_ids)) != len(selected_ids):
        raise ValueError("Topology requires unique sample IDs")
    n_samples = len(x)
    directory = None if output_dir is None else (
        Path(output_dir) / "absolute_core_metrics_v1" / "topology" / inputs.run_name)
    provenance = dict(run_name=inputs.run_name, split="test", representation="mu",
        selected_indices=indices.tolist(),
        sample_ids=None if selected_ids is None else selected_ids.tolist(),
        n_samples=n_samples, normalization=NORMALIZATION,
        sample_alignment="checked against parent X and mu arrays",
        ripser_config={**RIPSER_CONFIG, "thresh": "inf"},
        package_versions={name: _dependency_version(name) for name in ("ripser", "persim", "gudhi")},
        infinity_policy="Retain essential bars in raw diagrams; compare finite bars",
        wasserstein_convention=dict(backend="persim.wasserstein", order=1, ground_metric="L2"),
        wasserstein_w2_convention=dict(backend="gudhi.wasserstein.wasserstein_distance",
            order=2, ground_metric="L2", keep_essential_parts=False),
        bottleneck_convention=dict(backend="gudhi.bottleneck_distance", e=0.0, ground_metric="L-infinity"),
        config=vars(config), diagnostic_failures=[])

    def save_facts():
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "provenance.json").write_text(
                json.dumps(provenance, indent=2, allow_nan=False) + "\n")
            if "distance_scale_diagnostics" in provenance:
                pd.DataFrame([provenance["distance_scale_diagnostics"]]).to_csv(
                    directory / "distance_scale_diagnostics.csv", index=False)

    def unavailable(status, reason):
        provenance.update(status=status, warning=reason)
        save_facts()
        return pd.DataFrame.from_records(_undefined_values(status, reason, n_samples))

    if n_samples < 2:
        return unavailable("insufficient_samples", "At least two aligned observations are required")
    try:
        dx, dm = validate_paired_dissimilarity_matrices(pairwise_distances(x), pairwise_distances(mu))
    except ValueError as error:
        return unavailable("invalid_distances", str(error))
    scales = compute_distance_scale_diagnostics(dx, dm)
    provenance["distance_scale_diagnostics"] = scales
    if dx.max() <= config.diameter_epsilon or dm.max() <= config.diameter_epsilon:
        return unavailable("zero_diameter", "Input or mean-space diameter is numerically negligible")
    if ripser is None:
        return unavailable("missing_dependency", "ripser is required")
    cache_dir = None if output_dir is None else Path(output_dir) / "metrics_artifacts" / "topology_input_cache"
    input_diagrams, cache_key = _cached_input_diagrams(dx, selected_ids, cache_dir)
    provenance["input_cache_key"] = cache_key
    diagrams = compute_persistence_diagrams_h0_h1(dx, dm, dx.max(), dm.max(), input_diagrams)
    for name, diagram in diagrams.items():
        _finite_diagram(diagram)
        if "h1" in name and np.isinf(diagram).any():
            raise ValueError("Unexpected essential H1 class in full finite Rips filtration")
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            np.save(directory / (name + ".npy"), diagram)
    # Export main results before optional diagnostics: their failure must never
    # discard valid diagrams/distances. Missing dependencies have separate status.
    results = pd.concat([compute_normalized_wasserstein_h0_h1(diagrams, n_samples),
                         compute_normalized_wasserstein_w2_h0_h1(diagrams, n_samples),
                         compute_normalized_bottleneck_h0_h1(diagrams, n_samples)], ignore_index=True)
    provenance["status"] = "ok" if results.status.eq("ok").all() else "partial"
    save_facts()
    if directory is not None:
        results.to_csv(directory / "scalar_metrics.csv", index=False)

    def diagnostic(name, function):
        try:
            function()
        except Exception as error:
            provenance["diagnostic_failures"].append(
                dict(artifact=name, error_type=type(error).__name__, message=str(error)))
            warnings.warn(f"Topology diagnostic {name} failed: {error}", RuntimeWarning)

    if directory is not None:
        diagnostic("bar_summary", lambda: summarize_persistence_diagrams(diagrams).to_csv(
            directory / "persistence_bar_summary.csv", index=False))
        diagnostic("entropy", lambda: compute_persistence_entropy(diagrams, n_samples).to_csv(
            directory / "exploratory_entropy.csv", index=False))
        def save_betti():
            for name, values in compute_betti_curve_artifacts(diagrams).items():
                if isinstance(values, np.ndarray):
                    np.save(directory / (name + ".npy"), values)
            (directory / "betti_convention.json").write_text(json.dumps(
                dict(h0="reduced; essential component excluded", grid="exact shared event grid")))
        diagnostic("betti", save_betti)
        if config.ph_visualizations_enabled:
            def save_plot():
                # Display finite bars; essential H0 is retained in raw arrays.
                from matplotlib.figure import Figure
                figure = Figure(figsize=(8, 4))
                for dim, axis in enumerate(figure.subplots(1, 2)):
                    for space in ("input", "latent"):
                        points = _finite_diagram(diagrams[f"topology_{space}_h{dim}_persistence_diagram"])
                        axis.scatter(points[:, 0], points[:, 1], label=space, s=12)
                    axis.plot([0, 1], [0, 1], color="grey", linewidth=.5)
                    axis.set(xlabel="Birth", ylabel="Death", title=f"H{dim} (finite bars)")
                    axis.legend()
                figure.tight_layout()
                figure.savefig(directory / "persistence_diagrams.png", dpi=150)
            diagnostic("diagram_plot", save_plot)
    if config.landscapes_enabled:
        landscape_dir = None if output_dir is None else Path(output_dir) / "metrics_artifacts" / inputs.run_name / "landscape"
        diagnostic("landscape", lambda: run_landscape_metrics(
            diagrams, n_samples, landscape_dir, provenance,
            LandscapeConfig(config.landscape_num_landscapes, config.landscape_resolution)))
    save_facts()
    return results.reset_index(drop=True)
