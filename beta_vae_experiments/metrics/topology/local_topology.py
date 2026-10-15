"""Library-defined local homology of aligned test observations.

This module evaluates local homological structure around corresponding
observations. It is distinct from local-neighborhood preservation metrics such
as trustworthiness/continuity and from globally computed persistent homology.

Local persistence diagrams are produced by the public giotto-tda local-homology
implementation. The project does not define its own local complex or
local-homology estimator. KNeighborsLocalVietorisRips supplies both neighborhoods
and annulus coning; its kNN API avoids a project-defined cross-space radius.
The (1, 7) starting setting comes from the official local_homology tutorial,
not an assertion of optimality. No scaling, threshold or dimension shift occurs.

Bottleneck/Wasserstein distances, when reported, compare corresponding
library-generated local persistence diagrams and are not transformed into a
project-defined preservation score. Here only GUDHI Bottleneck is reported,
including essential bars. It is scale-sensitive, not a topological-type test.
No scalar aggregation is returned. H0 is the backend's reduced H0; H1 can
describe local branches, not just loops of the original unmodified space.

Sources (exact links and quantity definitions in docs/local_topology/
SCIENTIFIC_PROVENANCE.md): giotto-tda 0.6.2, public transformer and official
local_homology / local_hom_NLP_disambiguation notebooks; Tauzin et al., JMLR
2021; Jakubowski, Gašić & Zibrowius, Topology of Word Embeddings (arXiv:2011.09413), cited
by the NLP notebook; Boissonnat & Pritam, SoCG 2020, cited by the transformer.
Skraba & Wang, Approximating Local Homology from Samples, SODA 2014, is related
foundational literature, not a claimed exact attribution of giotto's kNN rule.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING
import hashlib
import inspect
import json
import warnings

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from beta_vae_experiments.metric_pipeline import CoreMetricInputs

SCALAR_COLUMNS = ('metric_name', 'value', 'status', 'warning', 'n_samples', 'split')
IDENTITY_COLUMNS = ('run_name', 'sample_id', 'selected_test_index')
DIAGRAM_COLUMNS = (*IDENTITY_COLUMNS, 'space', 'homology_dimension', 'birth', 'death', 'essential')
ANCHOR_COLUMNS = (*IDENTITY_COLUMNS, 'homology_dimension', 'bottleneck_distance', 'status', 'warning')
DESCRIPTION_COLUMNS = (*IDENTITY_COLUMNS, 'space', 'homology_dimension', 'status', 'warning')
VERIFIED_VERSION = '0.6.2'
VERIFIED_SOURCE_SHA256 = '7e5b5a547e0eeb6cc951c047dcfd326b0646f60fafaf4750a635d00f26e897f4'


@dataclass(frozen=True)
class LocalTopologyMetricConfig:
    """One sensitivity setting, matching the actual public giotto API.

    coeff is deliberately absent: the local transformer does not expose it.
    The verified backend fixes coeff=2. Changing it requires a new API audit.
    """
    enabled: bool = True
    n_neighbors: tuple[int, int] = (1, 7)
    homology_dimensions: tuple[int, ...] = (0, 1)
    collapse_edges: bool = False
    n_jobs: int | None = 1

    def __post_init__(self):
        object.__setattr__(self, 'n_neighbors', tuple(self.n_neighbors))
        object.__setattr__(self, 'homology_dimensions', tuple(self.homology_dimensions))
        if type(self.enabled) is not bool or type(self.collapse_edges) is not bool:
            raise TypeError('enabled and collapse_edges must be bool')
        if (len(self.n_neighbors) != 2 or
                any(type(k) is not int for k in self.n_neighbors) or
                not 1 <= self.n_neighbors[0] < self.n_neighbors[1]):
            raise ValueError('Require two integers 1 <= inner < outer; no automatic correction')
        if (not self.homology_dimensions or
                any(type(q) is not int or q < 0 for q in self.homology_dimensions) or
                tuple(sorted(set(self.homology_dimensions))) != self.homology_dimensions):
            raise ValueError('homology_dimensions must be sorted unique nonnegative integers')
        if self.n_jobs is not None and (type(self.n_jobs) is not int or self.n_jobs == 0):
            raise ValueError('n_jobs must be a nonzero integer or None')


def _version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return 'unavailable'


def _verified_transformer(config):
    """Verify source and API, never patch or approximate a missing implementation."""
    from gtda.local_homology import KNeighborsLocalVietorisRips
    import gtda.local_homology.simplicial as implementation
    source = Path(inspect.getfile(implementation)).read_bytes()
    if (_version('giotto-tda') != VERIFIED_VERSION or
            hashlib.sha256(source).hexdigest() != VERIFIED_SOURCE_SHA256):
        raise RuntimeError('Installed local-homology version/source differs from verified giotto-tda 0.6.2')
    return KNeighborsLocalVietorisRips(
        metric='euclidean', n_neighbors=config.n_neighbors,
        homology_dimensions=config.homology_dimensions,
        collapse_edges=config.collapse_edges, n_jobs=config.n_jobs,
    )


def diagram_in_dimension(triples, dimension):
    """Remove giotto diagonal padding only; preserve every positive-length bar."""
    triples = np.asarray(triples, dtype=float)
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError('Expected a giotto (n_features, 3) diagram')
    if (np.isnan(triples).any() or not np.isfinite(triples[:, [0, 2]]).all() or
            np.isneginf(triples[:, 1]).any() or
            np.any(triples[:, 1] < triples[:, 0])):
        raise ValueError('Invalid birth/death/dimension in library output')
    return triples[(triples[:, 2] == dimension) & (triples[:, 1] > triples[:, 0]), :2]


def compare_local_diagrams(diagram_x, diagram_mu):
    """Direct extended Bottleneck distance, including +inf death coordinates."""
    from gudhi import bottleneck_distance
    value = float(bottleneck_distance(diagram_x, diagram_mu, e=0.0))
    if np.isnan(value) or value < 0:
        raise ValueError('GUDHI returned an invalid Bottleneck distance')
    return value


def _validate_alignment(inputs, selected_indices, x_selected, mu_selected):
    indices = np.asarray(selected_indices)
    x, mu = np.asarray(x_selected, dtype=float), np.asarray(mu_selected, dtype=float)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError('selected_indices must be a one-dimensional integer array')
    if x.ndim != 2 or mu.ndim != 2 or not x.shape[1] or not mu.shape[1]:
        raise ValueError('X and mu must be 2D arrays with nonempty feature axes')
    if len(indices) != len(x) or len(x) != len(mu):
        raise ValueError('selected_indices, X and mu must contain the same aligned anchors')
    if not np.isfinite(x).all() or not np.isfinite(mu).all():
        raise ValueError('X and mu must be finite')
    if (len(np.unique(indices)) != len(indices) or np.any(indices < 0) or
            np.any(indices >= len(inputs.x_test)) or np.any(indices >= len(inputs.mu_test))):
        raise ValueError('Indices must be unique valid test-row positions')
    if (not np.array_equal(x, np.asarray(inputs.x_test)[indices]) or
            not np.array_equal(mu, np.asarray(inputs.mu_test)[indices])):
        raise ValueError('Selected arrays must equal X_test[indices] and mu_test[indices] in order')
    if inputs.sample_ids_test is None:
        ids = indices.astype(str)
    else:
        all_ids = np.asarray(inputs.sample_ids_test)
        if all_ids.ndim != 1 or len(all_ids) != len(inputs.x_test) or pd.isna(all_ids).any():
            raise ValueError('sample_ids_test must be nonmissing and aligned to test rows')
        ids = all_ids[indices].astype(str)
    if len(set(ids)) != len(ids):
        raise ValueError('Selected sample IDs must be unique')
    return indices, x, mu, ids


def run_local_topology_metrics_pipeline(
    inputs: CoreMetricInputs, selected_indices: np.ndarray,
    x_selected: np.ndarray, mu_selected: np.ndarray,
    config: LocalTopologyMetricConfig, output_dir: Path | None = None,
) -> pd.DataFrame:
    """Persist full per-anchor descriptions/distances; return no Core scalars.

    With output_dir=None, artifacts are available in result.attrs['local_topology'].
    A different sensitivity configuration must use a distinct output_dir/run_name;
    changing an existing artifact configuration in place is rejected.
    Duplicate-containing spaces are conservatively uncomputed (tie ambiguity),
    without deduplication, jitter, new neighborhoods or a fabricated zero score.
    """
    result = pd.DataFrame(columns=SCALAR_COLUMNS)
    if not config.enabled:
        return result
    indices, x, mu, ids = _validate_alignment(inputs, selected_indices, x_selected, mu_selected)
    config_json = json.loads(json.dumps(asdict(config)))
    directory = None
    if output_dir is not None:
        directory = Path(output_dir) / 'absolute_core_metrics_v1/local_topology' / inputs.run_name
        old_config = directory / 'config.json'
        if old_config.exists() and json.loads(old_config.read_text()) != config_json:
            raise ValueError('Use a distinct artifact location for a different local-homology sensitivity setting')
        directory.mkdir(parents=True, exist_ok=True)

    parameters = dict(
        giotto_tda_version=_version('giotto-tda'),
        local_homology_transformer='gtda.local_homology.KNeighborsLocalVietorisRips',
        verified_source_sha256=VERIFIED_SOURCE_SHA256,
        distance_metric='euclidean', distance_normalization='none',
        homology_dimensions=list(config.homology_dimensions), coefficient_field=2,
        reduced_homology=True, representation='mu',
        backend='gtda.homology.VietorisRipsPersistence -> gph.ripser_parallel',
        giotto_ph_version=_version('giotto-ph'), sklearn_version=_version('scikit-learn'),
        diagram_distance_backend='gudhi.bottleneck_distance',
        gudhi_version=_version('gudhi'), bottleneck_e=0.0,
        essential_policy='retain all library-returned positive-infinite-death bars',
        padding_policy='remove exact birth=death triples only',
        sample_id_source='test_row_position' if inputs.sample_ids_test is None else 'sample_ids_test',
        transformer_parameters=dict(metric='euclidean', n_neighbors=list(config.n_neighbors),
            homology_dimensions=list(config.homology_dimensions),
            collapse_edges=config.collapse_edges, n_jobs=config.n_jobs),
        spaces={},
    )
    descriptions, diagram_rows, comparisons, raw_outputs = [], [], [], {}
    diagrams, states = {}, {}
    for space, points in [('x', x), ('mu', mu)]:
        status, warning, raw = 'ok', None, None
        if len(points) < config.n_neighbors[1]:
            status, warning = 'degenerate_input', 'Too few observations for configured outer neighborhood; no automatic k reduction'
        elif len(np.unique(points, axis=0)) != len(points):
            status, warning = 'degenerate_input', 'Duplicate points or complete collapse: strict adapter does not resolve tied duplicate neighborhoods'
        else:
            try:
                transformer = _verified_transformer(config)
            except (ImportError, RuntimeError, OSError) as error:
                status, warning = 'failed_source_verification', str(error)
            else:
                try:
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter('always')
                        transformer.fit(points)
                        backend = transformer.homology.get_params()
                        if (tuple(transformer.neighborhood_params) != config.n_neighbors or
                                backend['coeff'] != 2 or backend['reduced_homology'] is not True or
                                backend['infinity_values'] is not None or
                                backend['max_edge_length'] != np.inf):
                            raise RuntimeError('Library altered verified neighborhood/backend settings')
                        raw = transformer.transform(points)
                    warning = '; '.join(dict.fromkeys(str(w.message) for w in captured)) or None
                    if raw.ndim != 3 or raw.shape[0] != len(points) or raw.shape[2] != 3:
                        raise RuntimeError('Unexpected local-homology output shape')
                    for triples in raw:
                        if not set(triples[:, 2]).issubset(config.homology_dimensions):
                            raise RuntimeError('Unexpected homology dimensions in library output')
                        for q in config.homology_dimensions:
                            diagram_in_dimension(triples, q)
                    raw_outputs[space] = raw
                except (ValueError, FloatingPointError) as error:
                    status, warning, raw = 'degenerate_input', str(error), None
                except RuntimeError as error:
                    status, warning, raw = 'failed_source_verification', str(error), None
        states[space] = status, warning
        parameters['spaces'][space] = dict(status=status, warning=warning)
        for i, (sample_id, index) in enumerate(zip(ids, indices)):
            identity = dict(run_name=inputs.run_name, sample_id=sample_id, selected_test_index=int(index))
            for q in config.homology_dimensions:
                descriptions.append(dict(**identity, space=space, homology_dimension=q, status=status, warning=warning))
                if raw is not None:
                    diagram = diagram_in_dimension(raw[i], q)
                    diagrams[space, i, q] = diagram
                    for birth, death in diagram:
                        diagram_rows.append(dict(**identity, space=space, homology_dimension=q,
                            birth=float(birth), death=float(death), essential=bool(np.isposinf(death))))

    for i, (sample_id, index) in enumerate(zip(ids, indices)):
        for q in config.homology_dimensions:
            status = ('failed_source_verification' if any(s[0] == 'failed_source_verification' for s in states.values())
                      else 'degenerate_input' if any(s[0] != 'ok' for s in states.values()) else 'ok')
            warning = '; '.join(f'{space}: {w}' for space, (_, w) in states.items() if w) or None
            value = np.nan
            if status == 'ok':
                try:
                    value = compare_local_diagrams(diagrams['x', i, q], diagrams['mu', i, q])
                    if np.isposinf(value):
                        warning = (warning + '; ' if warning else '') + 'Extended Bottleneck distance is infinite (unmatched essential classes)'
                except ImportError as error:
                    status, warning = 'failed_source_verification', str(error)
                except (ValueError, RuntimeError) as error:
                    status, warning = 'degenerate_input', str(error)
            comparisons.append(dict(run_name=inputs.run_name, sample_id=sample_id,
                selected_test_index=int(index), homology_dimension=q,
                bottleneck_distance=value, status=status, warning=warning))

    artifacts = dict(
        diagrams=pd.DataFrame(diagram_rows, columns=DIAGRAM_COLUMNS),
        per_anchor=pd.DataFrame(comparisons, columns=ANCHOR_COLUMNS),
        descriptions=pd.DataFrame(descriptions, columns=DESCRIPTION_COLUMNS),
        parameters=parameters, raw_outputs=raw_outputs,
    )
    if directory is None:
        result.attrs['local_topology'] = artifacts
    else:
        (directory / 'config.json').write_text(json.dumps(config_json, indent=2) + '\n')
        (directory / 'parameters.json').write_text(json.dumps(parameters, indent=2, allow_nan=False) + '\n')
        np.save(directory / 'selected_indices.npy', indices)
        for key, filename in [('diagrams', 'local_topology_diagrams.csv'),
                              ('per_anchor', 'local_topology_per_anchor.csv'),
                              ('descriptions', 'local_topology_description_status.csv')]:
            artifacts[key].to_csv(directory / filename, index=False)
        for space in ('x', 'mu'):
            path = directory / f'local_topology_raw_{space}.npy'
            if space in raw_outputs:
                np.save(path, raw_outputs[space])
            elif path.exists():
                path.unlink()  # Do not leave a stale success artifact after a failed rerun.
    return result
