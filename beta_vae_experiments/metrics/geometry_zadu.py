"""Official ZADU core metrics for input-to-latent geometry preservation.

Scientific scope
----------------
This module asks whether pairwise and neighborhood relationships in the
preprocessed encoder input ``X`` survive in a VAE representation.  It does not
claim that the input geometry is biologically ideal; preprocessing, feature
selection, scaling, and Euclidean distance define the reference geometry.

The implementation intentionally delegates metric computation to the public
ZADU 0.4.2 API:

``ZADU(specifications, original_space).measure(embedded_space)``

No project-local distance matrix, nearest-neighbor ranking, co-ranking matrix,
or metric formula is substituted.  This makes the numerical estimator and its
shared prerequisite scheduling traceable to:

Jeon et al. (2023), *ZADU: A Python Library for Evaluating the Reliability of
Dimensionality Reduction Embeddings*, IEEE VIS,
DOI: 10.1109/VIS54172.2023.00048.

Two VAE representations are evaluated independently by the surrounding
pipeline:

* ``mu`` is the deterministic posterior mean.  Posterior means are standard
  point embeddings for downstream dimensionality reduction, for example in
  scVI (Lopez et al., Nature Methods 2018,
  DOI: 10.1038/s41592-018-0229-2) and bmVAE (Yan, Ma, and Yu,
  Bioinformatics 2023, DOI: 10.1093/bioinformatics/btac790).
* ``z`` is the sample-ID-aligned realized posterior draw stored by Varix,
  ``z = mu + sigma * epsilon``.  It additionally exposes distortion caused by
  posterior uncertainty.  Sampled VAE representations have been used for
  cancer multi-omics integration by Hira et al. (Scientific Reports
  2021, DOI: 10.1038/s41598-021-85285-4).

The six requested ZADU measure families are:

* TNC: trustworthiness and continuity;
* MRRE: false-neighbor and missing-neighbor mean relative rank preservation;
* LCMC: neighborhood overlap corrected for overlap expected by chance;
* ND: dissimilarity between shared-nearest-neighbor graph structures;
* SN_STRESS: pairwise-distance stress after optimal global rescaling;
* NM_STRESS: non-metric stress after monotone isotonic distance fitting.

TNC, MRRE, and LCMC are higher-is-better. ND and both stress measures are
lower-is-better. The metrics must remain separate; averaging them would mix
different scales, directions, and scientific questions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from zadu import MEASURE, ZADU, make_spec


@dataclass(frozen=True)
class ZADUGeometryResult:
    """Typed result of one official ZADU input-versus-embedding evaluation.

    Attributes:
        trustworthiness:
            Preservation of embedded neighbors.  A high value means points that
            appear close in the embedding were generally also close in the
            input; false neighbors or *intrusions* reduce the score.
        continuity:
            Preservation of input neighbors.  A high value means points that
            were close in the input generally remain close in the embedding;
            missing neighbors or *extrusions* reduce the score.
        mrre_false:
            ZADU's rank-sensitive preservation score for false embedded
            neighbors.  Unlike TNC, MRRE weights relative rank disagreement.
        mrre_missing:
            ZADU's rank-sensitive preservation score for input neighbors
            missing from the embedded neighborhood.
        lcmc:
            Local Continuity Meta-Criterion at the selected ``k``.  It equals
            observed k-neighborhood overlap minus the random-overlap baseline
            ``k / (n - 1)``.  Identity therefore gives ``1 - k/(n-1)``, not
            exactly one, and poor embeddings may produce negative values.
        neighbor_dissimilarity:
            Difference between the input and embedding shared-nearest-neighbor
            graphs at ``k``. Zero means identical shared-neighbor structure;
            larger values indicate stronger structural disagreement.
        scale_normalized_stress:
            ZADU stress after fitting one global scale factor between the two
            pairwise-distance matrices.  Zero means exact preservation up to
            global scaling.
        non_metric_stress:
            Kruskal-style stress after fitting a monotone relationship between
            original and embedded distances.  Zero means perfect preservation
            of distance ordering, even if absolute distances are rescaled or
            monotonically transformed.
    """

    trustworthiness: float
    continuity: float
    mrre_false: float
    mrre_missing: float
    lcmc: float
    neighbor_dissimilarity: float
    scale_normalized_stress: float
    non_metric_stress: float


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
) -> ZADUGeometryResult:
    """Compute the label-free geometry core through ZADU's public class API.

    Args:
        input_space:
            Matrix ``(n_samples, n_input_features)`` containing the aligned,
            filtered, and scaled encoder input.  Row ``i`` must represent the
            same sample as row ``i`` of ``latent_space``.
        latent_space:
            Matrix ``(n_samples, latent_dimension)`` containing either
            posterior means ``mu`` or persisted posterior draws ``z``.
        k:
            Neighborhood size for TNC, MRRE, and LCMC.  This is a reporting
            scale, not a fitted model parameter.  ZADU requires
            ``1 <= k < n_samples``.

    Returns:
        The eight scalar outputs produced by six official ZADU measure
        specifications.  The result order is decoded explicitly from the same
        specification order below.

    Raises:
        ValueError:
            If the matrices are not finite 2D arrays, sample rows are not
            aligned in count, fewer than three samples are available, or ``k``
            is invalid.

    Notes:
        ZADU internally computes and shares Euclidean pairwise distances,
        nearest neighbors, and full rankings required by these measures.  This
        function performs only input-contract validation and typed unpacking;
        it does not reimplement any metric or geometric prerequisite.

        Example for ``n=40`` and ``k=10``: an identity embedding has TNC and
        MRRE scores of 1, both stresses of 0, and
        ``LCMC = 1 - 10/39 ~= 0.74359`` because LCMC subtracts chance overlap.
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
            f"ZADU k must satisfy 1 <= k <= {max_k} for n_samples={n_samples}, got {k}"
        )
    specifications = [
        make_spec(MEASURE.TNC, k=k),
        make_spec(MEASURE.MRRE, k=k),
        make_spec(MEASURE.LCMC, k=k),
        make_spec(MEASURE.ND, k=k),
        make_spec(MEASURE.SN_STRESS),
        make_spec(MEASURE.NM_STRESS),
    ]
    raw_results = ZADU(
        specifications,
        x,
        return_local=False,
        verbose=False,
        geodesic=False,
    ).measure(embedding)

    if len(raw_results) != len(specifications):
        raise RuntimeError(
            "ZADU returned an unexpected number of measure-family results"
        )
    tnc, mrre, lcmc, nd, scale_stress, non_metric_stress = raw_results
    values = {
        "trustworthiness": float(tnc["trustworthiness"]),
        "continuity": float(tnc["continuity"]),
        "mrre_false": float(mrre["mrre_false"]),
        "mrre_missing": float(mrre["mrre_missing"]),
        "lcmc": float(lcmc["lcmc"]),
        "neighbor_dissimilarity": float(nd["neighbor_dissimilarity"]),
        "scale_normalized_stress": float(scale_stress["scale_normalized_stress"]),
        "non_metric_stress": float(non_metric_stress["non_metric_stress"]),
    }
    return ZADUGeometryResult(**values)


__all__ = ["ZADUGeometryResult", "compute_zadu_geometry"]
