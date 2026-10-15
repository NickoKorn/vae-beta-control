"""Encoder/latent geometry; no decoder call, reconstruction or training.

Jacobian spectra measure sensitivity, not curvature (Rifai et al., ICML 2011).
Actual smooth immersion curvature uses the Euclidean Gauss equation, separately.
A compressive encoder's J.T@J is semidefinite, not a full input Riemannian metric.
See encoder_geometry.md for definitions, assumptions and primary references.
"""
from dataclasses import dataclass
from pathlib import Path
import copy
import json
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from ..graph.graph_builder import build_graph


@dataclass(frozen=True)
class EncoderGeometryConfig:
    enabled: bool = False
    max_samples: int = 128
    jacobian_samples: int = 8
    k_values: tuple = (5, 10, 20)
    quadruples: int = 1024
    random_state: int = 31
    rank_rtol: float = 1e-7

    def __post_init__(self):
        for name in ('max_samples', 'jacobian_samples', 'quadruples'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.max_samples < 4:
            raise ValueError('max_samples must be at least four')
        if (not self.k_values or len(set(self.k_values)) != len(self.k_values)
                or any(type(k) is not int or k < 1 for k in self.k_values)):
            raise ValueError('k_values must be distinct positive integers')
        if not np.isfinite(self.rank_rtol) or not 0 < self.rank_rtol < 1:
            raise ValueError('rank_rtol must lie between zero and one')


def jacobian_spectrum(jacobian, rtol=1e-7):
    """All nonzero singular values via the small output Gram matrix; no p×p G."""
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.ndim != 2 or not np.isfinite(jacobian).all():
        raise ValueError('Jacobian must be a finite matrix')
    eigenvalues = np.linalg.eigvalsh(jacobian @ jacobian.T)
    singular = np.sqrt(np.maximum(eigenvalues, 0))[::-1]
    # Eigenvalues below squared tolerance may be dominated by Gram roundoff.
    cutoff = max(rtol, np.sqrt(np.finfo(float).eps)) * singular[0] if len(singular) else 0
    positive = singular[singular > cutoff]
    probabilities = positive / positive.sum() if len(positive) else positive
    return dict(rank=len(positive), input_nullity=jacobian.shape[1]-len(positive),
                operator_norm=float(singular[0]) if len(singular) else 0.,
                frobenius_squared=float(np.square(jacobian).sum()),
                effective_rank=float(np.exp(-np.sum(probabilities*np.log(probabilities)))) if len(positive) else 0.,
                condition_nonzero=float(positive[0]/positive[-1]) if len(positive) else np.nan,
                full_input_metric_positive_definite=len(positive) == jacobian.shape[1]), singular


def immersion_curvature(immersion, point, *, smooth, rank_rtol=1e-7):
    """Actual intrinsic curvature of a C² full-rank immersion into Euclidean space.

    point parametrizes a known chart. This is NOT automatically the TCGA manifold.
    Sectional value refers to the first two metric-orthonormal frame directions.
    Scalar curvature is the full trace, with S² of radius R having scalar 2/R².
    """
    result = dict(status='unavailable', scalar_curvature=np.nan, sectional_01=np.nan,
                  second_fundamental_form_squared=np.nan)
    if not smooth:
        return dict(result, reason='C2_regularity_not_established')
    if point.ndim != 1 or len(point) < 2:
        return dict(result, reason='chart_dimension_must_be_at_least_two')
    jacobian = torch.func.jacrev(immersion)(point)
    if jacobian.ndim != 2:
        raise ValueError('Immersion must map one coordinate vector to one ambient vector')
    gram = jacobian.T @ jacobian
    values, vectors = torch.linalg.eigh(gram)
    cutoff = max(rank_rtol**2, torch.finfo(point.dtype).eps * 10) * values[-1]
    if not torch.isfinite(values).all() or values[0] <= cutoff or values[-1] <= 0:
        return dict(result, reason='rank_deficient_pullback')
    hessian = torch.func.jacrev(torch.func.jacrev(immersion))(point)
    flat = hessian.flatten(1)
    normal = flat - jacobian @ torch.linalg.solve(gram, jacobian.T @ flat)
    normal = normal.reshape(hessian.shape)
    frame = vectors / torch.sqrt(values)[None, :]
    second = torch.einsum('aij,ik,jl->akl', normal, frame, frame)
    diagonal = torch.diagonal(second, dim1=1, dim2=2)
    scalar = diagonal.sum(1).square().sum() - second.square().sum()
    sectional = (second[:,0,0]*second[:,1,1]).sum() - second[:,0,1].square().sum()
    return dict(status='ok', reason=None, scalar_curvature=float(scalar.detach()),
                sectional_01=float(sectional.detach()),
                second_fundamental_form_squared=float(second.square().sum().detach()))


def sampled_hyperbolicity(distances, quadruples):
    """Four-point delta lower bound on the observed finite metric, not curvature."""
    d = np.asarray(distances, dtype=float)
    q = np.asarray(quadruples, dtype=int)
    if (d.ndim != 2 or d.shape[0] != d.shape[1] or not np.isfinite(d).all()
            or (d < 0).any() or not np.allclose(d,d.T) or not np.allclose(np.diag(d),0)):
        raise ValueError('Require finite symmetric nonnegative distances with zero diagonal')
    if q.ndim != 2 or q.shape[1] != 4 or (q < 0).any() or (q >= len(d)).any():
        raise ValueError('Invalid quadruples')
    a,b,c,e = q.T
    sums = np.sort(np.stack((d[a,b]+d[c,e], d[a,c]+d[b,e], d[a,e]+d[b,c])), axis=0)
    delta = (sums[2]-sums[1])/2
    diameter = d.max()
    return dict(delta_lower_bound=float(delta.max()), delta_mean=float(delta.mean()),
                delta_over_diameter=float(delta.max()/diameter) if diameter > 0 else np.nan)


def latent_geometry(x, mu, sample_ids, config):
    """Bounded paired cohort: raw scale, angular shape, kNN path approximations."""
    x, mu, ids = np.asarray(x, float), np.asarray(mu, float), np.asarray(sample_ids)
    if (x.ndim != 2 or mu.ndim != 2 or len(x) != len(mu) or len(ids) != len(x)
            or ids.ndim != 1 or pd.isna(ids).any() or len(set(map(str,ids))) != len(ids)
            or not np.isfinite(x).all() or not np.isfinite(mu).all()):
        raise ValueError('Require finite paired X/mu and unique aligned IDs')
    if len(x) < 4 or len(x) > config.max_samples:
        raise ValueError('Use a paired cohort of 4..max_samples rows')
    rows, distance_rows = [], []
    rng = np.random.default_rng(config.random_state)
    quadruples = np.array([rng.choice(len(x), 4, replace=False) for _ in range(config.quadruples)])
    input_distances = squareform(pdist(x))
    triangle = np.triu_indices(len(x),1)

    def record(name, value, status='ok', warning=None, k=None):
        finite = np.isfinite(value)
        rows.append(dict(metric_name=name, value=float(value) if finite else np.nan,
                         status=status if finite else 'unavailable', warning=warning,
                         n_samples=len(x), split='test', k=k))

    for name, values in [('input',x), ('mu',mu)]:
        norms = np.linalg.norm(values,axis=1)
        for statistic, value in [('mean',norms.mean()),('std',norms.std()),('q10',np.quantile(norms,.1)),('q90',np.quantile(norms,.9))]:
            record(f'{name}_norm_{statistic}', value)
        record(f'{name}_zero_norm_fraction',np.mean(norms == 0))
        record(f'{name}_unique_row_fraction',len(np.unique(values,axis=0))/len(values))
        record(f'{name}_variance_trace',np.var(values,axis=0).sum())
        euclidean = input_distances if name == 'input' else squareform(pdist(values))
        for metric,value in sampled_hyperbolicity(euclidean,quadruples).items():
            record(f'{name}_{metric}',value, status='sampled_lower_bound')
        # Do not drop zero rows: an angular distance is undefined for such a cohort.
        if (norms == 0).any():
            record(f'{name}_angular_preservation_spearman',np.nan,
                   warning='Zero-norm vectors; no angular distance or silent row deletion')
        else:
            unit = values/norms[:,None]
            angular = np.arccos(np.clip(unit @ unit.T,-1,1));np.fill_diagonal(angular,0)
            reference = input_distances[triangle]
            rho = spearmanr(reference,angular[triangle]).statistic if np.ptp(reference)>0 and np.ptp(angular[triangle])>0 else np.nan
            record(f'{name}_angular_preservation_spearman',rho)
        for k in config.k_values:
            if k >= len(values):
                record(f'{name}_knn_path_to_chord_median_k{k}',np.nan,warning='k >= sample count',k=k)
                continue
            graph = build_graph(values,ids,k)
            # Reuse canonical union edges and their original distances. Never use
            # unit Forman weights as Riemannian edge lengths or invent bridges.
            edges = list(graph.undirected.edges(data='distance'))
            record(f'{name}_knn_edge_density_k{k}',2*len(edges)/(len(values)*(len(values)-1)),k=k)
            a = np.array([e[0] for e in edges]+[e[1] for e in edges])
            b = np.array([e[1] for e in edges]+[e[0] for e in edges])
            weights = np.array([e[2] for e in edges]*2)
            sparse = csr_matrix((weights,(a,b)),shape=(len(values),len(values)))
            paths = shortest_path(sparse,directed=False)[triangle]
            chords = euclidean[triangle]
            valid = np.isfinite(paths) & (chords>0)
            record(f'{name}_knn_connected_pair_fraction_k{k}',np.mean(np.isfinite(paths)),k=k)
            record(f'{name}_knn_path_to_chord_median_k{k}',np.median(paths[valid]/chords[valid]) if valid.any() else np.nan,
                   status='graph_path_approximation', k=k)
    latent_distances = squareform(pdist(mu))[triangle]
    reference = input_distances[triangle]
    positive = reference>0
    record('mu_to_input_distance_ratio_median',np.median(latent_distances[positive]/reference[positive]) if positive.any() else np.nan)
    record('mu_distance_preservation_spearman',spearmanr(reference,latent_distances).statistic
           if np.ptp(reference)>0 and np.ptp(latent_distances)>0 else np.nan)
    distance_rows = pd.DataFrame(dict(sample_id=ids.astype(str), input_norm=np.linalg.norm(x,axis=1), mu_norm=np.linalg.norm(mu,axis=1)))
    return pd.DataFrame(rows), distance_rows


def encoder_geometry(model, x, sample_ids, config, reference_mu=None, reference_logvar=None):
    """Checkpoint-dependent differential geometry of deterministic mu and q(z|x).

    Clone/eval the model, preserve original precision/mode/state. Work on float64
    CPU to stabilize spectra. Never call forward(), decode() or reparameterize().
    """
    model = copy.deepcopy(model).cpu().double().eval()
    records, spectra = [], []
    for row, (values, sample_id) in enumerate(zip(x,sample_ids)):
        point = torch.as_tensor(values,dtype=torch.float64)
        def parameters(t):
            mu, lv = model.encode(t[None])
            return torch.stack((mu[0],lv[0]))
        outputs = parameters(point).detach().numpy()
        if reference_mu is not None and not np.allclose(outputs[0],reference_mu[row],rtol=1e-4,atol=1e-5):
            raise ValueError('Checkpoint mu does not match stored posterior/preprocessing')
        if reference_logvar is not None and not np.allclose(outputs[1],reference_logvar[row],rtol=1e-4,atol=1e-5):
            raise ValueError('Checkpoint logvar does not match stored posterior/preprocessing')
        jacobian = torch.func.jacrev(parameters,chunk_size=1)(point).detach().numpy()
        j_mu, j_lv = jacobian
        sigma = np.exp(.5*outputs[1])
        # Euclidean pullback of diagonal Gaussian (mu,sigma) = W2 infinitesimal metric.
        # Fisher pullback is different: J_mu/sigma and J_logvar/sqrt(2).
        for name, matrix in [('mu',j_mu), ('posterior_w2',np.vstack((j_mu,.5*sigma[:,None]*j_lv))),
                             ('posterior_fisher',np.vstack((j_mu/sigma[:,None],j_lv/np.sqrt(2))))]:
            stats, singular = jacobian_spectrum(matrix,config.rank_rtol)
            records.append(dict(sample_id=str(sample_id),representation=name,**stats))
            spectra.extend(dict(sample_id=str(sample_id),representation=name,index=i,singular_value=float(s)) for i,s in enumerate(singular))
    return pd.DataFrame(records), pd.DataFrame(spectra)


def run_encoder_geometry_metrics_pipeline(inputs, selected_indices, x_selected, mu_selected,
                                          config, output_dir=None, model=None):
    """Optional adapter for existing scalar table; missing checkpoints stay explicit."""
    if not config.enabled:
        return pd.DataFrame()
    positions = np.sort(np.random.default_rng(config.random_state).choice(
        len(selected_indices), min(len(selected_indices),config.max_samples), replace=False))
    indices = np.asarray(selected_indices)[positions]
    x, mu = np.asarray(x_selected)[positions],np.asarray(mu_selected)[positions]
    ids_all = inputs.sample_ids_test
    if ids_all is None:
        raise ValueError('Encoder geometry requires real aligned sample IDs')
    ids = np.asarray(ids_all)[indices]
    if not np.array_equal(np.asarray(inputs.x_test)[indices],x) or not np.array_equal(np.asarray(inputs.mu_test)[indices],mu):
        raise ValueError('Encoder geometry rows disagree with declared selection')
    metrics, norms = latent_geometry(x,mu,ids,config)
    stats = singular = pd.DataFrame()
    if model is None:
        metrics.loc[len(metrics)] = dict(metric_name='encoder_mu_operator_norm_median',value=np.nan,
                                         status='missing_checkpoint',warning='No model available; latent-only evaluation',n_samples=0,split='test',k=None)
    else:
        count = min(config.jacobian_samples,len(x))
        positions = np.sort(np.random.default_rng(config.random_state).choice(len(x),count,replace=False))
        lv = np.asarray(inputs.logvar_test)[indices][positions] if inputs.logvar_test is not None else None
        stats, singular = encoder_geometry(model,x[positions],ids[positions],config,mu[positions],lv)
        for representation, group in stats.groupby('representation'):
            for name in ('rank','input_nullity','operator_norm','frobenius_squared','effective_rank','condition_nonzero'):
                metrics.loc[len(metrics)] = dict(metric_name=f'encoder_{representation}_{name}_median',value=group[name].median(),status='ok',warning=None,n_samples=count,split='test',k=None)
    metrics['split'] = inputs.metadata.get('evaluation_split','test')
    metrics.loc[len(metrics)] = dict(metric_name='encoder_intrinsic_scalar_curvature',value=np.nan,
                                     status='not_defined',warning='No smooth full-rank input chart established; J_E.T J_E may be singular; no ridge inverse or fake curvature',n_samples=len(x),split=inputs.metadata.get('evaluation_split','test'),k=None)
    if output_dir is not None:
        directory=Path(output_dir)/'metrics_artifacts'/inputs.run_name/'encoder_geometry'
        directory.mkdir(parents=True,exist_ok=True)
        metrics.to_parquet(directory/'metrics.parquet',index=False)
        norms.to_parquet(directory/'norms.parquet',index=False)
        stats.to_parquet(directory/'jacobian_profiles.parquet',index=False)
        singular.to_parquet(directory/'jacobian_spectra.parquet',index=False)
        (directory/'provenance.json').write_text(json.dumps(dict(config=vars(config),selected_indices=indices.tolist(),sample_ids=ids.astype(str).tolist(),
            representation='deterministic encoder mu and diagonal q(z|x); no sampled z',
            decoder_used=False,model_available=model is not None, graph_curvature='existing separate graph pipeline',
            distance_scope='bounded aligned sample; kNN paths are not exact Riemannian geodesics'),indent=2)+'\n')
    return metrics
