"""Geometry-preservation pipeline using official ZADU distortion measures.

The reference space is the sample-ID-aligned, preprocessed test input selected
by the main metric pipeline. The primary and only screening embedding is the
deterministic posterior mean ``mu_test``.

Primary geometry evaluation compares the preprocessed model input X with the
deterministic posterior-mean representation mu. ZADU measures quantify
different forms and granularities of representation distortion between these
two aligned point clouds. The extended set is temporarily used for metric
screening; redundant measures may be removed after cross-beta, cross-seed, and
cross-latent-dimension analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from zadu import MEASURE, ZADU, make_spec


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from beta_vae_experiments.metric_pipeline import CoreMetricInputs


@dataclass(frozen=True)
class GeometryMetricConfig:
    """Aligned X→mu geometry screening and deterministic sampling controls."""

    distance_metric: str = "euclidean"
    selected_k: int = 20
    max_samples: int = 3000
    random_state: int = 31

    def __post_init__(self) -> None:
        if self.distance_metric != "euclidean":
            raise ValueError("geometry distance_metric must be 'euclidean'")
        if self.selected_k < 1:
            raise ValueError("geometry selected_k must be positive")
        if self.max_samples < 3:
            raise ValueError("geometry max_samples must be at least three")

import dcor

def calc_distance_correlation(input: np.ndarray, mu: np.ndarray) -> float:
    """Return multivariate distance correlation for two aligned matrices."""
    input_values = np.asarray(input, dtype=np.float64)
    latent_values = np.asarray(mu, dtype=np.float64)
    if input_values.ndim != 2 or latent_values.ndim != 2:
        raise ValueError("input and mu must be two-dimensional matrices")
    if len(input_values) != len(latent_values):
        raise ValueError("input and mu must contain the same aligned samples")
    if not np.isfinite(input_values).all() or not np.isfinite(latent_values).all():
        raise ValueError("input and mu must contain only finite values")
    return float(dcor.distance_correlation(input_values, latent_values))

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


def _validated_space(value: np.ndarray, name: str) -> np.ndarray:
    """Return a finite two-dimensional float64 matrix without changing rows."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def compute_zadu_geometry(
    input_space: np.ndarray,
    latent_space: np.ndarray,
    k: int,
    sample_ids: np.ndarray | None = None,
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
        ``sigma`` and is not scale invariant. Related measures are not
        independent votes.
    Direction:
        Values are stored without inversion, clipping, or rescaling.

    In core metrics included: Temporarily, for cross-run metric screening.
    """
    x = _validated_space(input_space, "input_space")
    embedding = _validated_space(latent_space, "latent_space")
    if x.shape[0] != embedding.shape[0]:
        raise ValueError(
            "input_space and latent_space must contain the same aligned samples"
        )
    n_samples = x.shape[0]
    if n_samples < 3:
        raise ValueError("ZADU geometry requires at least three aligned samples")
    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    k = int(k)
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
        if pd.Index(aligned_ids).has_duplicates:
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
        make_spec(MEASURE.SNC, k=k),
        make_spec(MEASURE.TOPO, k=k),
        make_spec(MEASURE.DTM, sigma=0.1),
    ]

    warnings: list[str] = []
    failures: list[dict[str, Any]] = []
    raw_results: list[dict[str, Any]] = [{} for _ in specifications]
    local_columns: dict[str, np.ndarray] = {}

    for index, specification in enumerate(specifications):
        random_state = np.random.get_state()
        try:
            # ZADU SNC uses NumPy's global RNG. The seed is orchestration,
            # not a replacement or modification of the official measure.
            if str(specification["id"]) == str(MEASURE.SNC):
                np.random.seed(31)
            global_result, local_result = _run_zadu_measure(
                specification, x, embedding
            )
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
        finally:
            np.random.set_state(random_state)

    tnc, lcmc, mrre, nd, proc, srho, non_metric_stress, scale_stress, *tail = raw_results
    kl_0p01, kl_0p1, kl_1p0, snc, topo, dtm = tail

    def value(mapping: dict[str, object], key: str) -> float:
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

    return ZADUGeometryResult(
        global_metrics={
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
        },
        local_distortions=local_distortions,
        failed_measures=tuple(failures),
        warnings=tuple(warnings),
        metadata={"snc_effective_k": int(k)},
    )



def compute_distance_correlation(
    input_space: np.ndarray,
    latent_space: np.ndarray,
) -> dict[str, object]:
    """Compute multivariate distance correlation between aligned X and mu.

    Input representation:
        Sample-aligned, finite two-dimensional ``x_test`` and ``mu_test``
        subsets; feature dimensions may differ.
    """
    x = _validated_space(input_space, "input_space")
    mu = _validated_space(latent_space, "latent_space")
    if len(x) != len(mu):
        raise ValueError("input_space and latent_space must contain the same aligned samples")
    value = calc_distance_correlation(x, mu)
    finite = np.isfinite(value)
    return {
        "metric_name": "geometry_x_mu_distance_correlation",
        "value": value if finite else float("nan"),
        "status": "ok" if finite else "degenerate_input",
        "warning": None if finite else "Distance correlation is undefined.",
        "n_samples": len(x),
        "split": "test",
    }


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
    named_values = (
        (f"{prefix}_trustworthiness_zadu_k{k}", metrics["trustworthiness"]),
        (f"{prefix}_continuity_zadu_k{k}", metrics["continuity"]),
        (f"{prefix}_lcmc_zadu_k{k}", metrics["lcmc"]),
        (f"{prefix}_mrre_false_zadu_k{k}", metrics["mrre_false"]),
        (f"{prefix}_mrre_missing_zadu_k{k}", metrics["mrre_missing"]),
        (f"{prefix}_neighbor_dissimilarity_zadu_k{k}", metrics["neighbor_dissimilarity"]),
        (f"{prefix}_procrustes_zadu_k{k}", metrics["procrustes"]),
        (f"{prefix}_spearman_rho_zadu", metrics["spearman_rho"]),
        (f"{prefix}_non_metric_stress_zadu", metrics["non_metric_stress"]),
        (f"{prefix}_scale_normalized_stress_zadu", metrics["scale_normalized_stress"]),
        (f"{prefix}_kl_divergence_zadu_sigma_0p01", metrics["kl_divergence_sigma_0p01"]),
        (f"{prefix}_kl_divergence_zadu_sigma_0p1", metrics["kl_divergence_sigma_0p1"]),
        (f"{prefix}_kl_divergence_zadu_sigma_1p0", metrics["kl_divergence_sigma_1p0"]),
        (f"{prefix}_steadiness_zadu_k{k}", metrics["steadiness"]),
        (f"{prefix}_cohesiveness_zadu_k{k}", metrics["cohesiveness"]),
        (f"{prefix}_topographic_product_zadu_k{k}", metrics["topographic_product"]),
        (f"{prefix}_distance_to_measure_zadu_sigma_0p1", metrics["distance_to_measure_sigma_0p1"]),
    )
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
        "warning": (
            "; ".join(result.warnings)
            if is_degenerate
            else None
        ),
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
            Validated run data; only its run name is read here.
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
    x = np.asarray(x_selected, dtype=np.float64)
    mu = np.asarray(mu_selected, dtype=np.float64)
    n_samples = x.shape[0]
    if n_samples < 3:
        return pd.DataFrame(columns=(
            "metric_name", "value", "status", "warning", "n_samples", "split"
        ))
    selected_k = min(config.selected_k, (n_samples - 1) // 2)

    values = [
        compute_distance_correlation(x, mu),
    ]
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
    )
    values.extend(
        _metric_values(
            mu_result, representation="mu", k=selected_k, n_samples=n_samples
        )
    )
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
    "compute_distance_correlation",
    "compute_zadu_geometry",
    "run_geometry_metrics_pipeline",
]
