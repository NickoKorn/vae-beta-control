"""Geometry-preservation pipeline using official ZADU distortion measures.

The reference space is the sample-ID-aligned, preprocessed test input selected
by the main metric pipeline. The primary and only screening embedding is the
deterministic posterior mean ``mu_test``.

Primary geometry evaluation compares the preprocessed model input X with the
deterministic posterior-mean representation mu.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from importlib.metadata import version
from threading import Lock
import hashlib
import inspect
import json

from scipy.spatial.distance import pdist, squareform
from typing import Any

import numpy as np
import pandas as pd
import zadu as _zadu
from zadu import MEASURE, ZADU, make_spec
from zadu.measures import (
    local_continuity_meta_criteria, mean_relative_rank_error, neighbor_dissimilarity,
    procrustes, topographic_product, trustworthiness_continuity,
)


_RANK_MEASURES = {
    str(MEASURE.TNC): (trustworthiness_continuity, "ranking"),
    str(MEASURE.MRRE): (mean_relative_rank_error, "ranking"),
    str(MEASURE.LCMC): (local_continuity_meta_criteria, "knn"),
    str(MEASURE.ND): (neighbor_dissimilarity, "knn"),
    str(MEASURE.PROC): (procrustes, "knn"),
    str(MEASURE.TOPO): (topographic_product, "knn"),
}


def _self_first_neighbours(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Prepare ZADU neighbor/rank inputs: self=0, remaining ties by row index.

    Lee & Verleysen (2009), Section 2, p.1432, define distinct index-resolved
    ranks and explicitly set reflexive ranks to zero (text preceding Eq. 2):
    https://perso.uclouvain.be/michel.verleysen/papers/neurocomputing09jl.pdf
    Distances are float64; stable sorting implements the index convention.
    Tie order is not geometric information and depends on supplied row order.
    """
    distances = squareform(pdist(np.asarray(values, dtype=np.float64)))
    np.fill_diagonal(distances, -1.)
    order = np.argsort(distances, axis=1, kind="stable")
    return order[:, 1:k+1], np.argsort(order, axis=1, kind="stable")


def _run_rank_measure(measure_id, x, embedding, k, neighbours):
    """Use official ZADU formulas through its precomputed-neighbor arguments."""
    module, kind = _RANK_MEASURES[measure_id]
    (x_nn, x_rank), (mu_nn, mu_rank) = neighbours["x"], neighbours["mu"]
    options = (dict(knn_ranking_info=(x_nn, x_rank, mu_nn, mu_rank))
               if kind == "ranking" else dict(knn_info=(x_nn, mu_nn)))
    if "return_local" in inspect.signature(module.measure).parameters:
        options["return_local"] = True
    output = module.measure(x, embedding, k=k, **options)
    global_result, local_result = output if isinstance(output, tuple) else (output, None)
    return dict(global_result), None if local_result is None else dict(local_result)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from beta_vae_experiments.metric_pipeline import CoreMetricInputs


@dataclass(frozen=True)
class GeometryMetricConfig:
    """Parameter des unveränderten offiziellen ZADU-Screenings.

    k=20 ist unsere Referenz-Nachbarschaft, kein publiziertes Optimum. Die
    Sensitivitätsprüfung verwendet k=5/10/20/40 auf denselben Patienten.
    Quelle: Lee & Verleysen (2009), https://doi.org/10.1016/j.neucom.2008.12.017
    max_samples ist ein Budget; die gemeinsame Pipeline kann weiter begrenzen.
    random_state steuert Subsampling und den NumPy-Zufall des SNC-Adapters.
    In ZADU 0.4.2 ignoriert SNC das übergebene k und verwendet floor(sqrt(n)).
    """

    core_only: bool = False
    distance_metric: str = "euclidean"
    selected_k: int = 20
    k_values: tuple[int, ...] | None = None
    max_samples: int = 3000
    random_state: int = 31
    # Produktionsreferenz: k=20, gemeinsame euklidische X->mu-Stichprobe.
    # Bei TopologyMetricConfig(max_samples=1000) nutzt die Hauptpipeline
    # min(3000,1000)=1000; das tatsächliche n wird hier immer gespeichert.
    # k/Sample-Size-Piloten sind getrennte Analysen, kein Test-AUC-Tuning.

    def __post_init__(self) -> None:
        if self.k_values is not None:
            object.__setattr__(self, "k_values", tuple(self.k_values))
            if (not self.k_values or len(set(self.k_values)) != len(self.k_values)
                    or self.selected_k not in self.k_values
                    or any(type(k) is not int or k < 1 for k in self.k_values)):
                raise ValueError("k_values must be distinct positive integers including selected_k")
        if self.distance_metric != "euclidean":
            raise ValueError("geometry distance_metric must be 'euclidean'")
        if type(self.selected_k) is not int or self.selected_k < 1:
            raise ValueError("geometry selected_k must be positive")
        if type(self.max_samples) is not int or self.max_samples < 3:
            raise ValueError("geometry max_samples must be an integer >= three")
        if type(self.random_state) is not int or not 0 <= self.random_state <= 2**32-1:
            raise ValueError("geometry random_state must be an integer in [0, 2**32-1]")


_ZADU_RNG_LOCK = Lock()
_ZADU_SOURCE_ROOT = Path(_zadu.__file__).parent
_ZADU_SOURCE_HASHES = {
    str(path.relative_to(_ZADU_SOURCE_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(_ZADU_SOURCE_ROOT.rglob("*.py"))
}
# Methodische Grundlagen; Formeln bleiben bei der gepinnten Bibliothek.
ZADU_SOURCES = {
    "implementation": "https://pypi.org/project/zadu/0.4.2/",
    "zadu_paper": "https://doi.org/10.1109/VIS54172.2023.00048",
    "rank_framework": "https://doi.org/10.1016/j.neucom.2008.12.017",
    "steadiness_cohesiveness": "https://doi.org/10.1109/TVCG.2021.3114833",
    "stress_scaling": "https://arxiv.org/abs/2408.07724",
    "topographic_product": "https://doi.org/10.1109/72.143371",
    "nonmetric_stress": "https://doi.org/10.1007/BF02289565",
}
# Was messen wir? Die Quellen stehen oben in ZADU_SOURCES.
#
# Trustworthiness: Kommen im Latentraum falsche Nachbarn hinzu? Höher ist besser.
# Continuity: Gehen ursprüngliche Nachbarn verloren? Höher ist besser.
# LCMC: Wie viele Nachbarn bleiben gemeinsam? Höher ist besser. Bei identischen
# Daten beträgt der Wert 1-k/(n-1), nicht 1. Negative Werte sind möglich.
# MRRE: Bleibt die Rangfolge der Nachbarn erhalten? ZADU gibt einen Gütewert
# zurück: 1 ist optimal, obwohl der Methodenname von einem Fehler spricht.
#
# Neighbor Dissimilarity: Wie stark ändert sich die Nachbarschaftsstruktur?
# Procrustes: Wie stark verformt sich die Geometrie lokaler Nachbarschaften?
# Bei beiden bedeutet ein kleinerer Wert weniger Veränderung.
#
# Spearman: Bleibt die Reihenfolge der paarweisen Abstände erhalten?
# Höher ist besser. ZADU verwendet auch die Nullen auf der Diagonale und zählt
# jedes Paar in beiden Richtungen; wir übernehmen diese Bibliothekskonvention.
#
# Nichtmetrischer Stress: Passen die Abstände nach einer Anpassung zusammen,
# die ihre Reihenfolge erhält? Skalennormierter Stress erlaubt stattdessen
# einen gemeinsamen Streckfaktor. Bei beiden ist kleiner besser.
# Topographic Product: Bewertet Nachbarschaftsverzerrung; ideal nahe 0.
# Positive und negative Werte sind möglich.
#
# Steadiness/Cohesiveness (SNC): Bleiben Gruppenzusammenhänge erhalten?
# Höher ist besser. ZADU 0.4.2 nutzt intern floor(sqrt(n)) Nachbarn und ignoriert
# unser übergebenes k. Ein k-Sweep ist deshalb kein SNC-Parametersweep.
#
# ZADU-KL/DTM: Vergleichen geglättete Dichtegewichte von X und mu.
# Kleiner ist besser. Das ist weder die VAE-Posterior-KL noch die klassische
# Distance-to-a-Measure-Funktion. sigma steuert die Glättung.
# ZADU teilt die Abstände jedes Raums durch dessen größten Abstand. Eine reine
# Größenänderung verschwindet dadurch; Samplezahl und Ausreißer bleiben relevant.
# Technisch: Gewicht = exp(-(Abstand/Durchmesser)**2 / sigma), einschließlich
# des eigenen Punkts; anschließend werden die Dichtegewichte auf Summe 1 normiert.
# Die bisherigen Ergebnisnamen bleiben erhalten.


def _run_zadu_measure(
    specification: dict[str, Any],
    input_space: np.ndarray,
    embedding: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run one official ZADU measure in its own fail-isolated object."""
    evaluator = ZADU(
        [specification],
        input_space,
        return_local=True,
        verbose=False,
        geodesic=False,
    )
    global_results, local_results = evaluator.measure(embedding)
    global_result = dict(global_results[0])
    local_result = local_results[0]
    return global_result, None if local_result is None else dict(local_result)


@dataclass(frozen=True)
class ZADUGeometryResult:
    """Stable result envelope for a variable ZADU screening metric set.

    In core metrics included: Yes.
    """

    # One raw global score per active ZADU output; values are never inverted.
    global_metrics: dict[str, float]
    # Pointwise outputs aligned to test sample IDs; not promoted to run scalars.
    local_distortions: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Per-measure failures preserve isolation when another ZADU metric succeeds.
    failed_measures: tuple[dict[str, Any], ...] = ()
    # Human-readable numerical or availability warnings for the run log.
    warnings: tuple[str, ...] = ()
    # Non-metric execution facts such as the effective S&C neighborhood size.
    metadata: dict[str, Any] = field(default_factory=dict)
    metric_statuses: dict[str, str] = field(default_factory=dict)
    metric_warnings: dict[str, str] = field(default_factory=dict)


def _validated_space(value: np.ndarray, name: str) -> np.ndarray:
    """Return a finite two-dimensional float64 matrix without changing rows."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) == 0:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def compute_zadu_geometry(
    input_space: np.ndarray,
    latent_space: np.ndarray,
    k: int,
    sample_ids: np.ndarray | None = None,
    *,
    random_state: int = 31,
    core_only: bool = False,
) -> ZADUGeometryResult:
    """Compute official measures through independent fail-isolated ZADU objects.

    Input representation:
        Sample-aligned preprocessed ``x_test`` and posterior means ``mu_test``.
    Metric family:
        Local and global geometry preservation.
    Mathematical definition:
        All screening measures use the definitions implemented by ZADU; this
        adapter only validates, schedules, extracts, and names their outputs.

    Scientific source:
        Jeon et al., ZADU, IEEE VIS 2023,
        DOI:10.1109/VIS54172.2023.00048.

    Repository provenance:
        Each specification is passed to a separate official ``zadu.ZADU``
        object. This intentionally favors public API usage and per-measure
        failure isolation over shared private preprocessing internals. No
        measure formula is implemented locally.
    Interpretation:
        Compares neighborhood and distance structure between aligned spaces.
    Limitations:
        Depends on distance, preprocessing, sample subset, and ``k``. ZADU KL
        divergence is a sensitivity metric because it depends on kernel scale
        ``sigma``; the pinned density implementation normalizes each diameter. Related measures are not
        independent votes.
    Direction:
        Values are stored without inversion, clipping, or rescaling.

    In core metrics included: Temporarily, for cross-run metric screening.
    """
    if version("zadu") != "0.4.2":
        raise RuntimeError("Geometry conventions are audited for zadu==0.4.2; re-audit before upgrading")
    x = _validated_space(input_space, "input_space")
    embedding = _validated_space(latent_space, "latent_space")
    if x.shape[0] != embedding.shape[0]:
        raise ValueError(
            "input_space and latent_space must contain the same aligned samples"
        )
    n_samples = x.shape[0]
    if n_samples < 3:
        raise ValueError("ZADU geometry requires at least three aligned samples")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    k = int(k)
    if type(random_state) is not int or not 0 <= random_state <= 2**32-1:
        raise ValueError("random_state must be an integer in [0, 2**32-1]")
    max_k = (n_samples - 1) // 2
    if not 1 <= k <= max_k:
        raise ValueError(
            f"ZADU k must satisfy 1 <= k <= {max_k} for "
            f"n_samples={n_samples}, got {k}"
        )
    if sample_ids is None:
        aligned_ids = np.asarray([str(index) for index in range(n_samples)])
    else:
        aligned_ids = np.asarray(sample_ids)
        if aligned_ids.ndim != 1 or len(aligned_ids) != n_samples:
            raise ValueError("sample_ids must be one-dimensional and aligned with X/mu")
        if pd.Index(aligned_ids).hasnans or pd.Index(aligned_ids).has_duplicates:
            raise ValueError("sample_ids must be unique")

    specifications = [
        make_spec(MEASURE.TNC, k=k),
        make_spec(MEASURE.LCMC, k=k),
        make_spec(MEASURE.MRRE, k=k),
        make_spec(MEASURE.ND, k=k),
        make_spec(MEASURE.PROC, k=k),
        make_spec(MEASURE.SRHO),
        make_spec(MEASURE.NM_STRESS),
        make_spec(MEASURE.SN_STRESS),
        make_spec(MEASURE.KL_DIV, sigma=0.01),
        make_spec(MEASURE.KL_DIV, sigma=0.1),
        make_spec(MEASURE.KL_DIV, sigma=1.0),
        # ZADU 0.4.2 accepts but ignores k here; actual floor(sqrt(n)) is recorded.
        # Keep upstream defaults explicit, without replacing its algorithm.
        make_spec(MEASURE.SNC, k=k, iteration=150, walk_num_ratio=0.3,
                  alpha=0.1, clustering_strategy="dbscan"),
        make_spec(MEASURE.TOPO, k=k),
        make_spec(MEASURE.DTM, sigma=0.1),
    ]

    warnings: list[str] = []
    failures: list[dict[str, Any]] = []
    raw_results: list[dict[str, Any]] = [{} for _ in specifications]
    local_columns: dict[str, np.ndarray] = {}

    space_facts = {}
    for name, space in (("x", x), ("mu", embedding)):
        distances = pdist(space)
        # FAISS neighbors use float32; collisions can occur even if float64
        # rows differ. No jitter/deduplication is performed to conceal them.
        with np.errstate(over="ignore", invalid="ignore"):
            float32 = space.astype(np.float32)
        # A tied nonzero distance is reported, not resolved with artificial noise.
        # Equal ranks can depend on backend/order even without duplicate points.
        dmatrix = squareform(distances)
        np.fill_diagonal(dmatrix, np.inf)
        sorted_distances = np.sort(dmatrix, axis=1)[:, :-1]
        tied_rows = int(np.sum(np.any(np.diff(sorted_distances, axis=1) == 0, axis=1)))
        unique = len(np.unique(space, axis=0))
        unique32 = len(np.unique(float32, axis=0))
        space_facts[name] = dict(n_features=space.shape[1], unique_rows=unique, distance_tie_rows=tied_rows,
            duplicate_rows=n_samples-unique, float32_unique_rows=unique32,
            float32_finite=bool(np.isfinite(float32).all()),
            diameter=float(np.max(distances)))
    if any(f["distance_tie_rows"] for f in space_facts.values()):
        warnings.append("Equal-distance ranks present; see recorded rank backend and tie policy; no jitter.")
    collapsed = any(f["diameter"] == 0 for f in space_facts.values())
    duplicate_neighbors = any(f["float32_unique_rows"] != n_samples for f in space_facts.values())
    invalid32 = any(not f["float32_finite"] for f in space_facts.values())
    invalid_distances = any(not np.isfinite(f["diameter"]) for f in space_facts.values())
    neighbours = None
    if duplicate_neighbors and not (collapsed or invalid_distances or invalid32):
        neighbours = dict(x=_self_first_neighbours(x, k), mu=_self_first_neighbours(embedding, k))
        warnings.append("Duplicate/float32-colliding rows: rank measures use self-first Float64 "
                        "neighbors via ZADU knn_info/knn_ranking_info; ties ordered by row index. "
                        "Tie order is a convention, not recovered geometric information.")
    for index, specification in enumerate(specifications):
        if core_only and index not in (0, 7, 11):
            continue
        measure_id = str(specification["id"])
        neighbor_measure = "k" in specification["params"]
        reason = None
        if collapsed or invalid_distances:
            reason = "A reference or embedding has zero/nonfinite diameter; preservation is undefined."
        elif invalid32 and neighbor_measure:
            reason = "Nonfinite Float32 coordinates; native ZADU neighbor search is undefined."
        elif duplicate_neighbors and neighbor_measure and measure_id not in _RANK_MEASURES:
            reason = "Duplicate/float32-colliding rows: this measure has no precomputed-neighbor interface."
        if reason:
            failures.append(dict(measure=str(specification["id"]), params=specification["params"],
                stage="input_guard", exception_type="DegenerateGeometry", message=reason))
            warnings.append(reason)
            continue
        try:
            # ZADU SNC uses NumPy's global RNG. The seed is orchestration,
            # not a replacement or modification of the official measure.
            if neighbours is not None and measure_id in _RANK_MEASURES:
                global_result, local_result = _run_rank_measure(measure_id, x, embedding, k, neighbours)
            elif measure_id == str(MEASURE.SNC):
                # Serialize our adapter calls; preserve the caller's RNG state.
                with _ZADU_RNG_LOCK:
                    caller_state = np.random.get_state()
                    try:
                        np.random.seed(random_state)
                        global_result, local_result = _run_zadu_measure(specification, x, embedding)
                    finally:
                        np.random.set_state(caller_state)
            else:
                global_result, local_result = _run_zadu_measure(specification, x, embedding)
            raw_results[index] = global_result
            for name, raw_values in (local_result or {}).items():
                local = np.asarray(raw_values)
                if local.ndim != 1 or len(local) != n_samples:
                    warnings.append(
                        f"ZADU local output {name!r} has shape {local.shape}; "
                        f"expected ({n_samples},) and was not persisted."
                    )
                    continue
                local_columns[str(name)] = local
        except Exception as metric_error:
            measure_id = str(specification["id"])
            failures.append(
                {
                    "measure": measure_id,
                    "params": dict(specification.get("params", {})),
                    "stage": "measure",
                    "exception_type": type(metric_error).__name__,
                    "message": str(metric_error),
                }
            )
            warnings.append(
                f"ZADU {measure_id} is undefined: "
                f"{type(metric_error).__name__}: {metric_error}"
            )

    tnc, lcmc, mrre, nd, proc, srho, non_metric_stress, scale_stress, *tail = raw_results
    kl_0p01, kl_0p1, kl_1p0, snc, topo, dtm = tail

    def value(mapping: dict[str, object], key: str) -> float:
        if core_only and not mapping:
            return float("nan")
        try:
            result = float(mapping[key])
        except (KeyError, TypeError, ValueError) as error:
            warnings.append(f"ZADU output {key!r} is unavailable: {error}")
            return float("nan")
        if not np.isfinite(result):
            warnings.append(f"ZADU output {key!r} is non-finite.")
            return float("nan")
        return result

    local_distortions = pd.DataFrame({"sample_id": aligned_ids})
    for name, local in local_columns.items():
        local_distortions[name] = local

    metrics = {
            "trustworthiness": value(tnc, "trustworthiness"),
            "continuity": value(tnc, "continuity"),
            "lcmc": value(lcmc, "lcmc"),
            "mrre_false": value(mrre, "mrre_false"),
            "mrre_missing": value(mrre, "mrre_missing"),
            "scale_normalized_stress": value(scale_stress, "scale_normalized_stress"),
            "neighbor_dissimilarity": value(nd, "neighbor_dissimilarity"),
            "procrustes": value(proc, "procrustes"),
            "spearman_rho": value(srho, "spearman_rho"),
            "non_metric_stress": value(non_metric_stress, "non_metric_stress"),
            "kl_divergence_sigma_0p01": value(kl_0p01, "kl_divergence"),
            "kl_divergence_sigma_0p1": value(kl_0p1, "kl_divergence"),
            "kl_divergence_sigma_1p0": value(kl_1p0, "kl_divergence"),
            "steadiness": value(snc, "steadiness"),
            "cohesiveness": value(snc, "cohesiveness"),
            "topographic_product": value(topo, "topographic_product"),
            "distance_to_measure_sigma_0p1": value(dtm, "distance_to_measure"),
        }
    if core_only:
        metrics = {name: metrics[name] for name in ("trustworthiness", "continuity",
            "steadiness", "cohesiveness", "scale_normalized_stress")}
    metadata = dict(core_only=core_only, geometry_schema_version=2, snc_effective_k=int(np.sqrt(n_samples)), snc_requested_k=k,
        snc_seed=random_state, snc_k_is_ignored_by_backend=True,
        specifications=specifications, distance_metric="euclidean", geodesic=False,
        n_samples=n_samples, spaces=space_facts,
        duplicate_policy="keep rows; self-first/index-resolved ranks via ZADU interfaces; SNC withheld; no jitter",
        rank_backend="precomputed_float64" if neighbours is not None else "native_zadu",
        rank_tie_policy="self rank zero; remaining exact ties by supplied row index" if neighbours is not None else "native backend",
        rank_adapted_measures=sorted(_RANK_MEASURES) if neighbours is not None else [],
        kernel_density=dict(formula="sum_j exp(-(D_ij/max(D))**2/sigma), then normalize weights",
            separate_diameter_normalization=True, includes_self=True,
            dtm_convention="L1 difference of kernel-density weights; not classical DTM"),
        versions={p: version(p) for p in ("zadu", "numpy", "scipy", "scikit-learn", "faiss-cpu")},
        sources=ZADU_SOURCES, zadu_source_sha256=_ZADU_SOURCE_HASHES,
        roles=dict(primary=["trustworthiness", "continuity", "lcmc"],
            kernel_outputs="exploratory ZADU density comparisons; not posterior information metrics"),
        backend_conventions=dict(snc_k="floor(sqrt(n)), supplied k ignored",
            lcmc_identity="1-k/(n-1)", mrre_optimum=1,
            spearman_pairs="full flattened distance matrices including diagonal",
            topographic_product_direction="signed; ideal near zero"))
    statuses = {key: "ok" if np.isfinite(v) else "degenerate_input" for key, v in metrics.items()}
    metric_warnings = {key: "Undefined ZADU output; see per-measure failure/provenance artifacts."
                       for key in metrics if statuses[key] != "ok"}
    return ZADUGeometryResult(global_metrics=metrics, local_distortions=local_distortions,
        failed_measures=tuple(failures), warnings=tuple(dict.fromkeys(warnings)), metadata=metadata,
        metric_statuses=statuses, metric_warnings=metric_warnings)





def _metric_values(
    result: ZADUGeometryResult,
    *,
    representation: str,
    k: int,
    n_samples: int,
) -> list[dict[str, object]]:
    """Map one typed ZADU result to stable, representation-explicit columns."""
    prefix = f"geometry_{representation}"
    metrics = result.global_metrics
    metric_names = {
        "trustworthiness": f"{prefix}_trustworthiness_zadu_k{k}",
        "continuity": f"{prefix}_continuity_zadu_k{k}",
        "lcmc": f"{prefix}_lcmc_zadu_k{k}",
        "mrre_false": f"{prefix}_mrre_false_zadu_k{k}",
        "mrre_missing": f"{prefix}_mrre_missing_zadu_k{k}",
        "neighbor_dissimilarity": f"{prefix}_neighbor_dissimilarity_zadu_k{k}",
        "procrustes": f"{prefix}_procrustes_zadu_k{k}",
        "spearman_rho": f"{prefix}_spearman_rho_zadu",
        "non_metric_stress": f"{prefix}_non_metric_stress_zadu",
        "scale_normalized_stress": f"{prefix}_scale_normalized_stress_zadu",
        "kl_divergence_sigma_0p01": f"{prefix}_kl_divergence_zadu_sigma_0p01",
        "kl_divergence_sigma_0p1": f"{prefix}_kl_divergence_zadu_sigma_0p1",
        "kl_divergence_sigma_1p0": f"{prefix}_kl_divergence_zadu_sigma_1p0",
        "steadiness": f"{prefix}_steadiness_zadu_k{k}",
        "cohesiveness": f"{prefix}_cohesiveness_zadu_k{k}",
        "topographic_product": f"{prefix}_topographic_product_zadu_k{k}",
        "distance_to_measure_sigma_0p1": f"{prefix}_distance_to_measure_zadu_sigma_0p1",
    }
    named_values = [(metric_names[key], value) for key, value in metrics.items()]
    values = []
    for name, raw_value in named_values:
        is_finite = bool(np.isfinite(raw_value))
        values.append(
            {"metric_name": name, "value": float(raw_value) if is_finite else float("nan"), "status": "ok" if is_finite else "degenerate_input", "warning": None
                    if is_finite
                    else (
                        f"ZADU metric {name!r} is undefined; see geometry warnings."
                    ), "n_samples": n_samples, "split": "test"}
        )
    return values


def _table_row(
    result: ZADUGeometryResult,
    *,
    run_name: str,
    representation: str,
    k: int,
    n_samples: int,
) -> dict[str, object]:
    """Create a readable long-form row without collapsing metric families."""
    metrics = result.global_metrics
    raw_values = tuple(metrics.values())
    is_degenerate = not all(np.isfinite(value) for value in raw_values)
    return {
        "run_name": run_name,
        "split": "test",
        "input_representation": "preprocessed_model_input_x_test",
        "latent_representation": representation,
        "n_samples": n_samples,
        "k": k,
        **metrics,
        "snc_effective_k": result.metadata["snc_effective_k"],
        "status": "degenerate_input" if is_degenerate else "ok",
        "warning": "; ".join(result.warnings) or None,
        "requested_k": result.metadata.get("requested_k", k),
        "k_was_clipped": result.metadata.get("k_was_clipped", False),
        "x_duplicate_rows": result.metadata["spaces"]["x"]["duplicate_rows"],
        "mu_duplicate_rows": result.metadata["spaces"]["mu"]["duplicate_rows"],
        "implementation": "one public zadu.ZADU object per isolated official measure",
    }


def run_geometry_metrics_pipeline(
    inputs: CoreMetricInputs,
    selected_indices: np.ndarray,
    x_selected: np.ndarray,
    mu_selected: np.ndarray,
    config: GeometryMetricConfig,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Evaluate the temporary label-free ZADU core-screening set on ``mu``.

    Input representation:
        The same deterministic sample rows of preprocessed ``x_test`` and
        posterior means ``mu_test``. ``z_test``, ``logvar_test``, and an
        explicit aggregated posterior are never used.

    Args:
        inputs:
            Parent arrays/IDs are checked locally; run name names the artifacts.
        selected_indices, x_selected, mu_selected:
            The aligned deterministic test subset selected by the main pipeline.
        config:
            Geometry configuration. ``selected_k`` is used for the ZADU
            neighborhood measures. ``distance_metric`` is currently required
            to be ``"euclidean"`` because that is the distance used internally
            by the selected official ZADU measures.

    Returns:
        Raw ZADU screening measures, including ZADU Spearman rho and Procrustes.

    Scientific interpretation:
        High trustworthiness, continuity, LCMC, steadiness, cohesiveness, and
        Spearman rho are usually favorable; low neighbor dissimilarity,
        Procrustes, stress, and KL divergence are usually favorable. Values are not
        inverted, clipped, rescaled, combined, or automatically selected.

    In core metrics included:
        Yes. This is the geometry group runner; metric selection determines
        which of its outputs enter the core result.
    """
    if config.k_values is not None:
        frames = []
        for k in (config.selected_k, *(k for k in config.k_values if k != config.selected_k)):
            if k >= len(x_selected) / 2:
                raise ValueError(f"Requested preservation k={k} requires more than {2*k} samples")
            destination = output_dir
            if destination is not None and k != config.selected_k:
                destination = Path(destination) / "sensitivity" / f"k{k}"
            frames.append(run_geometry_metrics_pipeline(
                inputs, selected_indices, x_selected, mu_selected,
                replace(config, selected_k=k, k_values=None), destination))
        # Global measures have no k suffix: retain their primary-k result once.
        return pd.concat(frames, ignore_index=True).drop_duplicates("metric_name", keep="first")
    x = _validated_space(x_selected, "x_selected")
    mu = _validated_space(mu_selected, "mu_selected")
    indices = np.asarray(selected_indices)
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or len(indices) != len(x) or len(mu) != len(x)
            or len(np.unique(indices)) != len(indices)
            or (indices < 0).any() or (indices >= len(inputs.x_test)).any()):
        raise ValueError("Geometry requires unique valid aligned selected_indices")
    if not np.array_equal(np.asarray(inputs.x_test)[indices], x):
        raise ValueError("Selected X does not match declared test rows")
    if not np.array_equal(np.asarray(inputs.mu_test)[indices], mu):
        raise ValueError("Selected mu does not match declared test rows")
    n_samples = x.shape[0]
    if n_samples < 3:
        return pd.DataFrame(columns=(
            "metric_name", "value", "status", "warning", "n_samples", "split"
        ))
    selected_k = min(config.selected_k, (n_samples - 1) // 2)

    values: list[dict[str, object]] = []
    selected_ids = None
    if inputs.sample_ids_test is not None:
        selected_ids = np.asarray(inputs.sample_ids_test)[
            np.asarray(selected_indices, dtype=int)
        ]
    mu_result = compute_zadu_geometry(
        x,
        mu,
        selected_k,
        sample_ids=selected_ids,
        random_state=config.random_state,
        core_only=config.core_only,
    )
    values.extend(
        _metric_values(
            mu_result, representation="mu", k=selected_k, n_samples=n_samples
        )
    )
    mu_result.metadata.update(requested_k=config.selected_k, effective_k=selected_k,
        k_was_clipped=selected_k != config.selected_k, configured_max_samples=config.max_samples,
        subset_seed=config.random_state, selected_indices=indices.tolist(),
        sample_ids=None if selected_ids is None else list(map(str, selected_ids)),
        parent_samples=len(inputs.x_test),
        array_sha256={name: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
                      for name, a in (("x", x), ("mu", mu))})
    row = _table_row(
        mu_result,
        run_name=inputs.run_name,
        representation="posterior_mean_mu_test",
        k=selected_k,
        n_samples=n_samples,
    )
    tables = {"geometry_summary": pd.DataFrame([row])}
    if len(mu_result.local_distortions.columns) > 1:
        tables["geometry_mu_zadu_local_distortions"] = mu_result.local_distortions
    if mu_result.failed_measures:
        tables["geometry_zadu_failed_measures"] = pd.DataFrame(
            mu_result.failed_measures
        )
    if output_dir is not None:
        directory = Path(output_dir) / "absolute_core_metrics_v1" / "geometry" / inputs.run_name
        directory.mkdir(parents=True, exist_ok=True)
        provenance = {**mu_result.metadata, "warnings": list(mu_result.warnings),
            "failed_measures": list(mu_result.failed_measures),
            "metric_statuses": mu_result.metric_statuses}
        # Manifest remains strict JSON even for a nonfinite distance diagnostic.
        for facts in provenance["spaces"].values():
            if facts["diameter"] is not None and not np.isfinite(facts["diameter"]):
                facts["diameter"] = None
        (directory / "geometry_provenance.json").write_text(
            json.dumps(provenance, indent=2, allow_nan=False) + "\n")
        pd.DataFrame.from_records(values).to_csv(directory / "scalar_metrics.csv", index=False)
        for name, table in tables.items():
            path = directory / f"{name}.csv"
            table.to_csv(path, index=False)
            print(path)
        indices_path = directory / "selected_test_indices.npy"
        np.save(indices_path, np.asarray(selected_indices, dtype=int))
        print(indices_path)
    return pd.DataFrame.from_records(values)


__all__ = [
    "ZADUGeometryResult",
    "compute_zadu_geometry",
    "run_geometry_metrics_pipeline",
]
