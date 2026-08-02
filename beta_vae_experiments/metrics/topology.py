"""Diameter-normalized Vietoris-Rips topology metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import (
    CoreMetricInputs,
    MetricGroupResult,
    MetricResult,
    MetricValue,
    SharedMetricContext,
    TopologyMetricConfig,
)

try:
    from ripser import ripser
except ImportError:  # pragma: no cover - dependency policy tested through status
    ripser = None

try:
    from persim import bottleneck, wasserstein
except ImportError:  # pragma: no cover
    bottleneck = None
    wasserstein = None


def normalize_metric_spaces_by_diameter(
    context: SharedMetricContext,
    epsilon: float = 1e-12,
) -> MetricResult:
    """Normalize input and latent metric spaces separately by diameter.

    Metric family:
        Topological data analysis.
    Space comparison:
        Input topology versus latent topology.
    Exact spaces:
        Aligned input metric space and posterior-mean latent metric space.
    Representations:
        Input and latent pairwise-distance matrices.
    Mathematical definition:
        D_X_norm=D_X/diam(X), D_mu_norm=D_mu/diam(mu), diam=max_ij D_ij.
    Scientific source:
        May, Krishnamoorthy, and Gambill, arXiv:2306.06727, Definition 2.5.
    Repository provenance:
        Custom implementation of the cited normalization.
    Interpretation:
        Removes independent positive scale from both metric spaces before PH.
    Limitations:
        Undefined for zero-diameter spaces and sensitive to outlying diameters.
    Direction:
        Descriptive only.
    """
    values = [
        MetricValue(
            "topology_input_diameter",
            float(context.input_diameter),
            n_samples=len(context.x_test_selected),
            split="test",
        ),
        MetricValue(
            "topology_latent_diameter",
            float(context.latent_diameter),
            n_samples=len(context.x_test_selected),
            split="test",
        ),
    ]
    if context.input_diameter <= epsilon or context.latent_diameter <= epsilon:
        warning = "At least one metric space has zero or negligible diameter."
        for value in values:
            value.status = "degenerate_input"
            value.warning = warning
        return MetricResult(values, warnings=[warning])
    return MetricResult(
        values,
        artifacts={
            "input_normalized_distances": context.input_distance_matrix
            / context.input_diameter,
            "latent_normalized_distances": context.latent_distance_matrix
            / context.latent_diameter,
            "topology_metric_normalization": "separate_diameter_normalization",
        },
    )


def compute_persistence_diagrams_h0_h1(
    normalized_input_distances: np.ndarray,
    normalized_latent_distances: np.ndarray,
) -> MetricResult:
    """Compute H0/H1 Vietoris-Rips persistence diagrams in both spaces.

    Metric family:
        Topological data analysis.
    Space comparison:
        Input only and latent topology only, stored as separate artifacts.
    Exact spaces:
        Separately diameter-normalized input and posterior-mean latent metric spaces.
    Representations:
        Input and latent persistence diagrams for H0 and H1.
    Mathematical definition:
        Vietoris-Rips persistence of the two precomputed normalized distance matrices.
    Scientific source:
        Munch, A User's Guide to Topological Data Analysis, 2017.
    Repository provenance:
        ripser.py ripser(..., distance_matrix=True, maxdim=1).
    Interpretation:
        H0 captures connectivity merges and H1 captures loop persistence.
    Limitations:
        Depends on metric, observations, and Vietoris-Rips filtration.
    Direction:
        Diagrams are descriptive artifacts.
    """
    if ripser is None:
        warning = "ripser is not installed."
        return MetricResult(
            [], warnings=[warning], artifacts={"status": "failed_source_verification"}
        )
    input_diagrams = ripser(normalized_input_distances, distance_matrix=True, maxdim=1)[
        "dgms"
    ]
    latent_diagrams = ripser(
        normalized_latent_distances, distance_matrix=True, maxdim=1
    )["dgms"]
    return MetricResult(
        [],
        artifacts={
            "input_h0": np.asarray(input_diagrams[0], dtype=np.float64),
            "input_h1": np.asarray(input_diagrams[1], dtype=np.float64),
            "latent_h0": np.asarray(latent_diagrams[0], dtype=np.float64),
            "latent_h1": np.asarray(latent_diagrams[1], dtype=np.float64),
        },
    )


def _diagram_frame(diagram: np.ndarray, space: str, dimension: int) -> pd.DataFrame:
    if diagram.size == 0:
        return pd.DataFrame(
            columns=[
                "birth",
                "death",
                "lifetime",
                "homology_dimension",
                "space",
                "is_essential",
            ]
        )
    essential = ~np.isfinite(diagram[:, 1])
    lifetime = diagram[:, 1] - diagram[:, 0]
    return pd.DataFrame(
        {
            "birth": diagram[:, 0],
            "death": diagram[:, 1],
            "lifetime": lifetime,
            "homology_dimension": dimension,
            "space": space,
            "is_essential": essential,
        }
    )


def compute_persistence_summary_descriptors(
    diagrams: dict[str, np.ndarray],
) -> MetricResult:
    """Summarize finite persistent features separately for each space and dimension.

    Metric family:
        Topological data analysis.
    Space comparison:
        Input only or latent topology only.
    Exact spaces:
        Individual H0/H1 diagrams of normalized input or posterior-mean latent spaces.
    Representations:
        Finite persistence-diagram lifetimes.
    Mathematical definition:
        Count, sum, maximum, and median of finite lifetimes death-birth.
    Scientific source:
        Munch, A User's Guide to Topological Data Analysis, 2017.
    Repository provenance:
        Custom summaries of ripser persistence diagrams.
    Interpretation:
        Characterizes persistent connectivity or loop features within one space.
    Limitations:
        Single-space summaries do not measure topology preservation.
    Direction:
        Descriptive only.
    """
    values: list[MetricValue] = []
    tables: dict[str, pd.DataFrame] = {}
    for space in ("input", "latent"):
        for dimension in (0, 1):
            key = f"{space}_h{dimension}"
            diagram = diagrams[key]
            frame = _diagram_frame(diagram, space, dimension)
            tables[key] = frame
            finite = frame.loc[~frame["is_essential"], "lifetime"].to_numpy(dtype=float)
            prefix = f"topology_{space}_h{dimension}"
            values.extend(
                [
                    MetricValue(f"{prefix}_finite_feature_count", len(finite)),
                    MetricValue(
                        f"{prefix}_total_persistence",
                        float(np.sum(finite)) if len(finite) else 0.0,
                    ),
                    MetricValue(
                        f"{prefix}_max_persistence",
                        float(np.max(finite)) if len(finite) else 0.0,
                    ),
                    MetricValue(
                        f"{prefix}_median_persistence",
                        float(np.median(finite)) if len(finite) else 0.0,
                    ),
                ]
            )
    return MetricResult(values, tables=tables)


def compute_betti_curves_h0_h1(
    diagrams: dict[str, np.ndarray],
    run_name: str,
    grid_points: int,
) -> MetricResult:
    """Compute H0/H1 Betti curves on a shared normalized filtration grid.

    Metric family:
        Topological data analysis.
    Space comparison:
        Input only and latent topology only as separate curves.
    Exact spaces:
        H0/H1 diagrams from normalized input and posterior-mean latent spaces.
    Representations:
        Input and latent Betti curves.
    Mathematical definition:
        beta_h(t)=number of intervals satisfying birth<=t<death.
    Scientific source:
        Munch, A User's Guide to Topological Data Analysis, 2017.
    Repository provenance:
        Custom vectorized implementation from the cited definition.
    Interpretation:
        Counts active connectivity or loop features over filtration.
    Limitations:
        Descriptive curve; not reduced to an arbitrary primary scalar.
    Direction:
        Descriptive only.
    """
    grid = np.linspace(0.0, 1.0, grid_points)
    rows = []
    for space in ("input", "latent"):
        for dimension in (0, 1):
            diagram = diagrams[f"{space}_h{dimension}"]
            for filtration in grid:
                if diagram.size:
                    count = int(
                        np.sum(
                            (diagram[:, 0] <= filtration) & (filtration < diagram[:, 1])
                        )
                    )
                else:
                    count = 0
                rows.append(
                    {
                        "run_name": run_name,
                        "space": space,
                        "homology_dimension": dimension,
                        "filtration": float(filtration),
                        "betti_number": count,
                    }
                )
    return MetricResult([], tables={"betti_curves": pd.DataFrame(rows)})


def compute_normalized_bottleneck_h0_h1(
    diagrams: dict[str, np.ndarray],
) -> MetricResult:
    """Compute H0/H1 bottleneck after metric-space diameter normalization.

    Metric family:
        Topological data analysis.
    Space comparison:
        Input topology versus latent topology.
    Exact spaces:
        H0/H1 diagrams of separately diameter-normalized input and posterior-mean latent spaces.
    Representations:
        Finite input and latent persistence diagrams.
    Mathematical definition:
        d_N(X,Z)=d_B(X/diam(X), Z/diam(Z)), separately for H0 and H1.
    Scientific source:
        May, Krishnamoorthy, and Gambill, arXiv:2306.06727, Definition 2.5.
    Repository provenance:
        persim.bottleneck on finite ripser diagrams after separate normalization.
    Interpretation:
        Scale-invariant discrepancy between persistent topological signatures.
    Limitations:
        Depends on metric/sample and emphasizes the largest unmatched feature.
    Direction:
        Lower is better.
    """
    if bottleneck is None:
        warning = "persim is not installed."
        return MetricResult(
            [
                MetricValue(
                    f"topology_h{dim}_normalized_bottleneck",
                    float("nan"),
                    "failed_source_verification",
                    warning,
                )
                for dim in (0, 1)
            ],
            warnings=[warning],
        )
    values = []
    for dimension in (0, 1):
        a = diagrams[f"input_h{dimension}"]
        b = diagrams[f"latent_h{dimension}"]
        a = a[np.isfinite(a).all(axis=1)] if a.size else np.empty((0, 2))
        b = b[np.isfinite(b).all(axis=1)] if b.size else np.empty((0, 2))
        if len(a) == 0 and len(b) == 0:
            distance = 0.0
        else:
            distance = float(bottleneck(a, b))
        values.append(
            MetricValue(f"topology_h{dimension}_normalized_bottleneck", distance)
        )
    return MetricResult(values)


def compute_normalized_wasserstein_h0_h1(
    diagrams: dict[str, np.ndarray],
) -> MetricResult:
    """Compute H0/H1 Wasserstein after metric-space diameter normalization.

    Metric family:
        Topological data analysis.
    Space comparison:
        Input topology versus latent topology.
    Exact spaces:
        H0/H1 diagrams of separately diameter-normalized input and posterior-mean latent spaces.
    Representations:
        Finite input and latent persistence diagrams.
    Mathematical definition:
        Wasserstein optimal-matching cost between the finite persistence
        diagrams of X/diam(X) and Z/diam(Z), separately for H0 and H1.
    Scientific source:
        Persistence-diagram Wasserstein distance; persim 0.3.8 documentation.
    Repository provenance:
        persim.wasserstein on finite ripser diagrams after separate normalization.
    Interpretation:
        Scale-invariant aggregate discrepancy across persistent topological features.
    Limitations:
        Depends on metric and sample and can accumulate many small or noisy
        feature discrepancies; essential infinite classes are excluded.
    Direction:
        Lower is better.
    """
    if wasserstein is None:
        warning = "persim is not installed."
        return MetricResult(
            [
                MetricValue(
                    f"topology_h{dim}_normalized_wasserstein",
                    float("nan"),
                    "failed_source_verification",
                    warning,
                )
                for dim in (0, 1)
            ],
            warnings=[warning],
        )
    values = []
    for dimension in (0, 1):
        a = diagrams[f"input_h{dimension}"]
        b = diagrams[f"latent_h{dimension}"]
        a = a[np.isfinite(a).all(axis=1)] if a.size else np.empty((0, 2))
        b = b[np.isfinite(b).all(axis=1)] if b.size else np.empty((0, 2))
        if len(a) == 0 and len(b) == 0:
            distance = 0.0
        else:
            distance = float(wasserstein(a, b, matching=False))
        values.append(
            MetricValue(f"topology_h{dimension}_normalized_wasserstein", distance)
        )
    return MetricResult(values)


def run_topology_metrics_pipeline(
    inputs: CoreMetricInputs,
    context: SharedMetricContext,
    config: TopologyMetricConfig,
) -> MetricGroupResult:
    """Compare persistent topology of aligned input and latent metric spaces.

    Intuition:
        This pipeline uses neither k-means nor a fixed k-nearest-neighbor graph.
        It starts from all pairwise distances in the shared sample subset and
        grows a Vietoris--Rips filtration by increasing the distance threshold.
        H0 records how initially separate patient components merge; H1 records
        one-dimensional loops that persist across distance scales.

        ``max_homology_dimension=1`` fixes the absolute core to H0 and H1.
        ``betti_grid_points`` only controls how finely the Betti curves are
        sampled: with 256 points over ``[0, filtration_max]``, the spacing is
        ``filtration_max / 255``. It does not alter persistence diagrams or
        bottleneck/Wasserstein distances. Separate diameter normalization maps
        each space to diameter one before comparing diagrams, removing overall
        scale but retaining relative distance and topological structure.

        ``max_samples`` is a computational cap. Because geometry and topology
        currently share one aligned context, the effective subset size is
        ``min(n_test, geometry.max_samples, topology.max_samples)``. Thus the
        defaults 3000 and 1000 make both families use at most 1000 observations.

    Metric family:
        Topological data analysis.
    Space comparison:
        Persistent H0/H1 structure of aligned preprocessed ``x_test`` versus
        posterior-mean ``mu_test``.
    Limitations:
        Results depend on preprocessing, distance metric, deterministic sample
        subset, diameter normalization, and finite-sample filtration. They do
        not identify biological meaning and do not replace neighbor-preservation
        metrics with known patient correspondence.
    """
    if len(context.x_test_selected) < 2:
        return MetricGroupResult(
            "topology",
            "degenerate_input",
            [],
            warnings=["Fewer than two observations."],
        )
    normalized = normalize_metric_spaces_by_diameter(context)
    if normalized.warnings:
        undefined = [
            MetricValue(
                f"topology_h{dim}_normalized_{metric}",
                float("nan"),
                "degenerate_input",
                normalized.warnings[0],
            )
            for dim in (0, 1)
            for metric in ("bottleneck", "wasserstein")
        ]
        return MetricGroupResult(
            "topology",
            "degenerate_input",
            normalized.values + undefined,
            artifacts=normalized.artifacts,
            warnings=normalized.warnings,
        )
    persistence = compute_persistence_diagrams_h0_h1(
        normalized.artifacts["input_normalized_distances"],
        normalized.artifacts["latent_normalized_distances"],
    )
    if persistence.artifacts.get("status") == "failed_source_verification":
        return MetricGroupResult(
            "topology",
            "failed_source_verification",
            normalized.values,
            warnings=persistence.warnings,
        )
    diagrams = {
        key: persistence.artifacts[key]
        for key in ("input_h0", "input_h1", "latent_h0", "latent_h1")
    }
    summaries = compute_persistence_summary_descriptors(diagrams)
    betti = compute_betti_curves_h0_h1(
        diagrams, inputs.run_name, config.betti_grid_points
    )
    bottleneck_comparison = compute_normalized_bottleneck_h0_h1(diagrams)
    wasserstein_comparison = compute_normalized_wasserstein_h0_h1(diagrams)
    comparison_warnings = (
        bottleneck_comparison.warnings + wasserstein_comparison.warnings
    )
    artifacts = {
        **diagrams,
        "topology_metric_normalization": "separate_diameter_normalization",
    }
    return MetricGroupResult(
        "topology",
        "ok" if not comparison_warnings else "failed_source_verification",
        normalized.values
        + summaries.values
        + bottleneck_comparison.values
        + wasserstein_comparison.values,
        tables={**summaries.tables, **betti.tables},
        artifacts=artifacts,
        warnings=comparison_warnings,
    )
