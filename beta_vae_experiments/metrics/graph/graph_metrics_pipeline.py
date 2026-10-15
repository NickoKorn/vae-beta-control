"""Fail-isolated graph families in the existing scalar-metric table format."""
from dataclasses import dataclass
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import json
import time
import warnings
import numpy as np
import pandas as pd
from beta_vae_experiments.metrics.graph.graph_builder import build_graph
from beta_vae_experiments.metrics.dimensionality.intrinsic_dimension import compute_global_id, compute_local_id
from beta_vae_experiments.metrics.graph.hubness_metrics import compute_hubness
from beta_vae_experiments.metrics.graph.graph_neighborhood_metrics import compute_neighborhood
from beta_vae_experiments.metrics.graph.graph_community_metrics import compute_communities
from beta_vae_experiments.metrics.graph.graph_connectivity_metrics import compute_connectivity
from beta_vae_experiments.metrics.graph.graph_path_metrics import compute_paths
from beta_vae_experiments.metrics.curvature.graph_curvature_metrics import compute_curvature
from beta_vae_experiments.metrics.graph.input_latent_graph_comparison import compare_graphs
from beta_vae_experiments.metrics.graph.graph_export import export_graph

FAMILY_OUTPUTS = {
    "global_id": ("global_id",),
    "local_id": tuple("local_id_"+s for s in ("mean", "median", "std", "q10", "q25", "q75", "q90")),
    "hubness": ("hubness_skewness", "antihub_fraction", "hub_occurrence", "hub_fraction", "robinhood", "groupie_ratio"),
    "neighborhood": ("reciprocity", "clustering", "transitivity", "degree_mean", "degree_std", "degree_min", "degree_max"),
    "community": ("communities", "modularity", "community_size_mean", "community_size_median", "community_size_min", "community_size_max", "largest_community_fraction", "community_entropy"),
    "connectivity": ("components", "giant_component_fraction", "isolated_nodes", "isolated_fraction", "lambda2"),
    "paths": ("lcc_average_path_length", "lcc_diameter", "global_efficiency", "betweenness_mean", "betweenness_median", "betweenness_q90", "betweenness_max"),
    **{kind: tuple(kind+"_"+s for s in ("mean", "median", "std", "q10", "q90", "negative_fraction")) for kind in ("forman", "ollivier")},
}


@dataclass(frozen=True)
class GraphMetricConfig:
    # None inherits geometry.selected_k, not a separately tuned parameter.
    k_values: tuple[int, ...] | None = None
    export_graphml: bool = False
    ollivier: bool = False
    max_path_nodes: int = 3000

    def __post_init__(self):
        if self.k_values is not None:
            object.__setattr__(self, "k_values", tuple(self.k_values))
        if self.k_values is not None and (not self.k_values or len(set(self.k_values)) != len(self.k_values)
                or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in self.k_values)):
            raise ValueError("k_values must be distinct positive integers")
        if self.max_path_nodes < 2:
            raise ValueError("max_path_nodes must be at least two")


def run_graph_metrics_pipeline(inputs, selected_indices, x_selected, mu_selected,
                               config, geometry, output_dir=None):
    """Use the existing geometry cohort; record every selected row and failure.

    Optional families never substitute estimators or change rows/k. Missing
    values use NaN/status/warning, as in existing dimensionality metrics.
    """
    start = time.perf_counter()
    versions = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "scikit-dimension",
                    "scikit-hubness", "networkx", "igraph", "GraphRicciCurvature", "python-louvain"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    indices = np.asarray(selected_indices)
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or len(np.unique(indices)) != len(indices) or (indices < 0).any()
            or (indices >= len(inputs.x_test)).any()):
        raise ValueError("Selected indices must be unique valid test row indices")
    if not np.array_equal(np.asarray(inputs.x_test)[indices], x_selected, equal_nan=True):
        raise ValueError("Selected X does not match the declared test row selection")
    if hasattr(inputs, "mu_test") and not np.array_equal(np.asarray(inputs.mu_test)[indices], mu_selected, equal_nan=True):
        raise ValueError("Selected mu does not match the declared test row selection")
    ids = None if inputs.sample_ids_test is None else np.asarray(inputs.sample_ids_test)[indices]
    directory = None if output_dir is None else Path(output_dir)/"metrics_artifacts"/inputs.run_name/"graph_metrics"
    rows, failures = [], []
    metadata = dict(package_versions=versions, analysis_seed=geometry.random_state,
                    selected_indices=indices.tolist(), n_samples=len(indices),
                    full_test_samples=len(inputs.x_test), run_metadata=dict(inputs.metadata),
                    cohort="existing geometry selection; no additional subsampling",
                    representation="preprocessed x_test and posterior mean mu_test",
                    distance_metric=geometry.distance_metric, config=vars(config),
                    k_values=list(config.k_values or (geometry.selected_k,)),
                    leiden_resolution=1.0, leiden_iterations=2, curvature="unit-weight Forman 1d",
                    path_scope="ASPL/diameter LCC; efficiency/betweenness whole union")
    metadata['unique_rows'] = {name: len(np.unique(values, axis=0)) for name, values in
                               (('input', x_selected), ('latent', mu_selected))}

    def record(metrics, prefix, k, warning=None):
        for name, value in metrics.items():
            finite = np.isfinite(value)
            if not finite:
                failures.append(dict(metric=name, representation=prefix, k=k,
                                     reason=warning or "Nonfinite/undefined output; no imputation"))
            rows.append(dict(metric_name=f"graph_{prefix}{name}_k{k}", value=float(value) if finite else np.nan,
                             status="ok" if finite else "unavailable", warning=warning if finite else warning or "Library returned nonfinite value; no imputation",
                             n_samples=len(indices), split="test", k=k))

    def run_family(name, call, prefix, k, metrics, nodes, edges):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = call()
            if isinstance(result, dict):
                result = (result, {}, {})
            elif len(result) == 2:
                result = (*result, {})
            vals, node_values, edge_values = result
            warning = "; ".join(dict.fromkeys(str(w.message) for w in caught)) or None
            record(vals, prefix, k, warning)
            metrics.update(vals); nodes.update(node_values); edges.update(edge_values)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            failures.append(dict(family=name, representation=prefix, k=k, reason=reason))
            missing = {key: np.nan for key in FAMILY_OUTPUTS.get(name, (name,))}
            metrics.update(missing)
            record(missing, prefix, k, reason)

    for k in metadata["k_values"]:
        spaces = []
        for prefix, values in (("input_", x_selected), ("latent_", mu_selected)):
            metrics, nodes, edges = {}, {}, {}
            try:
                if ids is None:
                    raise ValueError("Aligned sample IDs are required; positional IDs are not invented")
                graph = build_graph(values, ids, k, geometry.distance_metric)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                failures.append(dict(family="builder", representation=prefix, k=k, reason=reason))
                record({key: np.nan for family, keys in FAMILY_OUTPUTS.items()
                        if family != "ollivier" or config.ollivier for key in keys}, prefix, k, reason)
                continue
            families = [
                ("global_id", lambda: compute_global_id(values)),
                ("local_id", lambda: compute_local_id(values, k)),
                ("hubness", lambda: compute_hubness(graph, geometry.random_state)),
                ("neighborhood", lambda: compute_neighborhood(graph)),
                ("community", lambda: compute_communities(graph, geometry.random_state)),
                ("connectivity", lambda: compute_connectivity(graph, geometry.random_state)),
                ("paths", lambda: compute_paths(graph, config.max_path_nodes)),
                ("forman", lambda: compute_curvature(graph)),
            ]
            if config.ollivier:
                families.append(("ollivier", lambda: compute_curvature(graph, ollivier=True)))
            for name, call in families:
                run_family(name, call, prefix, k, metrics, nodes, edges)
            if directory is not None:
                run_family("export", lambda: (export_graph(graph, nodes, edges, directory/f"{prefix}k{k}", config.export_graphml,
                    getattr(inputs, "sample_metadata_test", None)) or {}), prefix, k, {}, {}, {})
            spaces.append((graph, metrics, nodes))
        if len(spaces) == 2:
            a, b = spaces
            run_family("comparison", lambda: compare_graphs(a[0], b[0], a[1], b[1], a[2], b[2]), "", k, {}, {}, {})
        else:
            missing = {'delta_'+key: np.nan for family, keys in FAMILY_OUTPUTS.items()
                       if family != 'ollivier' or config.ollivier for key in keys}
            missing.update(partition_ari=np.nan, partition_nmi=np.nan,
                           hub_identity_spearman=np.nan)
            record(missing, '', k, 'Comparison unavailable because graph construction failed')
    metadata.update(failures=failures, elapsed_seconds=time.perf_counter()-start)
    if directory is not None:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            (directory/"metadata.json").write_text(json.dumps(metadata, indent=2, default=str)+"\n")
        except OSError as exc:
            record({'metadata_artifact': np.nan}, '', 'all', str(exc))
            warnings.warn(f'Graph metadata artifact could not be written: {exc}', RuntimeWarning)
    frame = pd.DataFrame(rows)
    frame.attrs["graph_metadata"] = metadata
    return frame
