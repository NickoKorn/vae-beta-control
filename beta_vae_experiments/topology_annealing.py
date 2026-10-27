"""Sigma-unit topology, a validation-only H0 annealer and post-training analysis.

The AUTOENCODIX package and existing LinearBetaWarmup are not modified.
All new numerical methods and controller/analysis logic live in this file.
"""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def _posterior_arrays(mu, logvar):
    mu, logvar = np.asarray(mu, dtype=np.float64), np.asarray(logvar, dtype=np.float64)
    if mu.ndim != 2 or mu.shape != logvar.shape or min(mu.shape) == 0:
        raise ValueError('mu/logvar must have matching nonempty (samples, dimensions) shapes')
    if not np.isfinite(mu).all() or not np.isfinite(logvar).all():
        raise ValueError('Posterior parameters must be finite')
    return mu, logvar


def sigma_unit_distances(mu, logvar, block=256):
    """Pairwise mean separation in common posterior-standard-deviation units.

    Input representation: diagonal Gaussian patient posteriors, mu/logvar (N,d).
    Mathematical definition: sqrt(sum_k (mu_i-mu_j)^2 / mean(var_i,var_j)).
        The average variance is evaluated in log space without clipping.
    Scientific source: Gaussian Bhattacharyya mean term, e.g.
        https://github.com/dherrera1911/sqfa/blob/5d66578e9e4d497c58c6dbdd5af3b37789cfe04a/src/sqfa/distances.py
    Interpretation: d^2/8 is the mean term of Gaussian Bhattacharyya distance.
        exp(-d^2/8) omits its nonnegative covariance penalty: it is an upper
        bound on BC, exact only for equal covariance, not full overlap in general.
    Limitations: triangle inequality and separation of unequal-variance Gaussians
        are not guaranteed. Rips/single-linkage accept these dissimilarities,
        but a metric-ball/Nerve-theorem interpretation is not generally valid.
        Zero means identical means, NOT proof of posterior collapse to the prior.
        Invariant to axis permutations/signs and common scaling of mu and sigma
        (also coordinatewise scaling); not general rotations of diagonal models.
        Scores still depend on sample count, latent dimension and data selection.
    Direction: smaller means closer posterior means relative to uncertainty.
    In core metrics included: yes, independent of diameter-normalized TDA.
    """
    mu, logvar = _posterior_arrays(mu, logvar)
    if type(block) is not int or block < 1:
        raise ValueError('block must be a positive integer')
    distances = np.empty((len(mu), len(mu)), dtype=np.float64)
    for start in range(0, len(mu), block):
        stop = min(start+block, len(mu))
        log_average = np.logaddexp(logvar[start:stop,None,:], logvar[None,:,:])-np.log(2.)
        with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
            delta = np.abs(mu[start:stop,None,:]-mu[None,:,:])
            standardized = np.exp(np.log(delta)-.5*log_average)
            distances[start:stop] = np.sqrt(np.sum(standardized*standardized, axis=-1))
    if not np.isfinite(distances).all():
        raise ValueError('Nonfinite sigma-unit distances; no variance clipping is performed')
    np.fill_diagonal(distances, 0.)
    return distances


def h0_death_times(dist):
    """Single-linkage merge heights: N-1 finite H0 deaths/MST edge weights.

    Strict threshold convention: an edge merges before epsilon iff d < epsilon.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    """
    dist = np.asarray(dist, dtype=np.float64)
    if (dist.ndim != 2 or dist.shape[0] != dist.shape[1] or not len(dist)
            or not np.isfinite(dist).all() or np.any(dist < 0)
            or not np.allclose(dist, dist.T) or not np.allclose(np.diag(dist), 0)):
        raise ValueError('Distances must be finite, symmetric, nonnegative, square and zero-diagonal')
    return linkage(squareform(dist, checks=False), method='single')[:,2] if len(dist)>1 else np.empty(0)


def _eps_tag(epsilon):
    return f'{epsilon:g}'.replace('.', 'p')


def _summarize_distances(dist, eps):
    deaths = h0_death_times(dist)
    result = {}
    for epsilon in eps:
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('eps must contain positive finite thresholds')
        tag = _eps_tag(epsilon)
        # Fraction of merge events, not the fraction of patients with a neighbor.
        result['h0_frac_below_eps'+tag] = float(np.mean(deaths<epsilon)) if len(deaths) else np.nan
        result['h0_components_at_eps'+tag] = int(1+np.sum(deaths>=epsilon))
    nearest = dist.copy();np.fill_diagonal(nearest,np.inf)
    result.update(h0_max_gap_sigma=float(deaths.max()) if len(deaths) else 0.,
                  nn_median_sigma=float(np.median(nearest.min(axis=1))) if len(dist)>1 else np.nan)
    return result, deaths


def sigma_unit_h0_summary(mu, logvar, eps=(1.,2.,3.)):
    """H0 merge fractions and the existing epsilon-delta KL collapse criterion.

    Zero separation is necessary, not sufficient, for exact prior collapse.
    A coordinate is epsilon-delta collapsed when at least 99% of patients
    have KL_j < 0.01. This is independent of the H0 stopping thresholds.
    frac_below measures N-1 merge events, rather than individual patients.
    """
    mu, logvar = _posterior_arrays(mu, logvar)
    summary = _summarize_distances(sigma_unit_distances(mu,logvar), eps)[0]
    summary['n_collapsed_dims_kl'] = _collapsed_dimensions(mu, logvar)
    return summary


def _collapsed_dimensions(mu, logvar):
    from .metrics.information_theoretic import compute_conditional_kl_matrix
    from .metrics.latent_usage import PosteriorCollapseConfig
    policy = PosteriorCollapseConfig()
    kl = compute_conditional_kl_matrix(mu, logvar)
    return int(np.sum(np.mean(kl < policy.primary_epsilon, axis=0) >= 1-policy.delta))


@dataclass(frozen=True)
class SigmaTopologyConfig:
    eps: tuple[float,...] = (1.,2.,3.)
    max_samples: int = 1000
    compute_h1: bool = False

    def __post_init__(self):
        object.__setattr__(self,'eps',tuple(self.eps))
        if not self.eps or len(set(self.eps))!=len(self.eps) or any(not np.isfinite(e) or e<=0 for e in self.eps):
            raise ValueError('eps must be distinct positive finite values')
        if type(self.max_samples) is not int or self.max_samples<2:
            raise ValueError('max_samples must be an integer >=2')
        if type(self.compute_h1) is not bool:
            raise ValueError('compute_h1 must be boolean')


def plot_sigma_barcode(deaths, path, eps=(1.,2.,3.)):
    """Write the finite H0 barcode without diameter normalization."""
    import matplotlib.pyplot as plt
    deaths = np.sort(deaths)[::-1]
    fig, ax = plt.subplots(figsize=(9,4))
    ax.hlines(np.arange(len(deaths)), 0, deaths, color='tab:blue', linewidth=.6)
    for epsilon in eps: ax.axvline(epsilon,linestyle='--',label=f'epsilon={epsilon:g}')
    ax.set(xlabel='σ-Einheiten',ylabel='Finite H0 bars',title='H0: posterior-mean separation in sigma units')
    ax.legend();fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def _h1_summary(dist, eps):
    from ripser import ripser
    from .metrics.topology.preservation import RIPSER_CONFIG
    diagram = ripser(dist, **RIPSER_CONFIG)['dgms'][1]
    values = {'h1_alive_eps'+_eps_tag(e):int(np.sum((diagram[:,0]<e)&(diagram[:,1]>=e))) for e in eps}
    values.update(_main_h1_bar(diagram))
    return values, diagram


def _main_h1_bar(diagram):
    """Longest finite H1 bar; its identity as a ground-truth loop is not assumed."""
    finite = diagram[np.isfinite(diagram).all(axis=1) & (diagram[:,1]>diagram[:,0])]
    if not len(finite):
        return dict(h1_main_birth_sigma=np.nan,h1_main_death_sigma=np.nan,h1_main_persistence_sigma=0.)
    birth,death = finite[np.argmax(finite[:,1]-finite[:,0])]
    return dict(h1_main_birth_sigma=float(birth),h1_main_death_sigma=float(death),h1_main_persistence_sigma=float(death-birth))


def run_sigma_unit_topology_pipeline(inputs, config, output_dir=None, *, selected_indices=None):
    """Use the shared test subset, including its actual posterior log-variances."""
    indices = selected_indices
    if indices is None:
        n = len(inputs.mu_test)
        indices = (np.arange(n) if n<=config.max_samples else
                   np.sort(np.random.default_rng(31).choice(n,config.max_samples,replace=False)))
    indices = np.asarray(indices,dtype=int)
    names = list(_summarize_distances(np.zeros((2,2)),config.eps)[0])
    names.append('n_collapsed_dims_kl')
    if config.compute_h1:
        names += ['h1_alive_eps'+_eps_tag(e) for e in config.eps]
        names += list(_main_h1_bar(np.empty((0,2))))
    warning = None
    if inputs.logvar_test is None:
        values, status, warning = dict.fromkeys(names,np.nan), 'missing_posterior', 'Actual logvar unavailable'
    elif len(indices)<2:
        values, status, warning = dict.fromkeys(names,np.nan), 'insufficient_samples', 'At least two patients required'
    else:
        mu = np.asarray(inputs.mu_test)[indices];lv = np.asarray(inputs.logvar_test)[indices]
        dist = sigma_unit_distances(mu,lv)
        values, deaths = _summarize_distances(dist,config.eps)
        values['n_collapsed_dims_kl'] = _collapsed_dimensions(mu,lv)
        diagram = None
        if config.compute_h1:
            extra,diagram = _h1_summary(dist,config.eps);values.update(extra)
        status = 'ok'
        if output_dir is not None:
            folder=Path(output_dir)/'metrics_artifacts'/inputs.run_name/'topology_sigma/test'
            folder.mkdir(parents=True,exist_ok=True)
            np.save(folder/'h0_death_times.npy',deaths)
            if diagram is not None:np.save(folder/'h1_diagram.npy',diagram)
            plot_sigma_barcode(deaths,folder/'h0_barcode.png',config.eps)
            ids = getattr(inputs,'sample_ids_test',None)
            (folder/'provenance.json').write_text(json.dumps(dict(config=asdict(config),split='test',
                selected_indices=indices.tolist(),sample_ids=None if ids is None else np.asarray(ids)[indices].astype(str).tolist(),
                normalization='posterior variances; no diameter normalization',
                threshold='strict d < epsilon',zero_interpretation='coincident means, not sufficient evidence of collapse'),indent=2)+'\n')
    return pd.DataFrame([dict(metric_name='topology_sigma_'+key,value=float(value),status=status,
        warning=warning,n_samples=len(indices),split='test') for key,value in values.items()])


def _beta_by_epoch(run_folder):
    run_folder=Path(run_folder)
    for name,column in [('beta_schedule.csv','beta_config'),('controller_log.csv','effective_beta')]:
        path=run_folder/name
        if path.is_file():return pd.read_csv(path).set_index('epoch')[column].to_dict()
    path=run_folder/'epoch_losses.csv'
    if path.is_file():
        table=pd.read_csv(path)
        if 'effective_beta_factor' in table:
            return table.loc[table.split.eq('valid')].set_index('epoch').effective_beta_factor.to_dict()
    return {}


def sigma_topology_over_epochs(run_folder, split='valid', config=None):
    """Analyze saved posteriors with fixed IDs; missing parameters remain undefined.

    Native AUTOENCODIX _sigma_buffer stores latent_logvar, despite the name.
    Current NPZ exports call it logvar; legacy sigmas is the same log-variance.
    Never reinterpret it as a standard deviation or replace missing mu by zeros.
    """
    config=config or SigmaTopologyConfig();run_folder=Path(run_folder)
    if split not in ('train','valid'):raise ValueError('Snapshot analysis uses train or valid')
    rows=[];selected_ids=None;beta=_beta_by_epoch(run_folder)
    for path in sorted((run_folder/'epoch_snapshots').glob(f'epoch_*_{split}.npz')):
        epoch=int(path.stem.split('_')[1]);row=dict(epoch=epoch,effective_beta=beta.get(epoch,np.nan),split=split)
        with np.load(path) as saved:
            lv_key='logvar' if 'logvar' in saved else 'sigmas'
            if 'mu' not in saved or lv_key not in saved or 'sample_ids' not in saved:
                rows.append(dict(**row,status='missing_posterior',warning='Actual mu/logvar/IDs unavailable'));continue
            ids=saved['sample_ids'].astype(str)
            if selected_ids is None:
                indices=np.arange(len(ids)) if len(ids)<=config.max_samples else np.sort(np.random.default_rng(31).choice(len(ids),config.max_samples,False))
                selected_ids=ids[indices]
            positions=pd.Index(ids).get_indexer(selected_ids)
            if (positions<0).any() or len(np.unique(ids))!=len(ids):raise ValueError('Snapshot cohort IDs changed')
            summary=sigma_unit_h0_summary(saved['mu'][positions],saved[lv_key][positions],config.eps)
            if config.compute_h1:
                extra, _ = _h1_summary(sigma_unit_distances(saved['mu'][positions],saved[lv_key][positions]),config.eps)
                summary.update(extra)
            rows.append(dict(**row,n_samples=len(positions),status='ok',warning=None,**summary))
    table=pd.DataFrame(rows);table.to_csv(run_folder/f'sigma_topology_{split}_epochs.csv',index=False)
    return table


def betti_curves(dist, eps_grid=None):
    """Sample b0/b1 over scale; additionally retain exact diagram breakpoints.

    Default log grid [0.1,100] is an analysis range, not a universal scale.
    Standard Rips convention: birth <= epsilon < death (unlike legacy strict
    merge summaries). Finite point sampling can create short spurious H1 bars.
    """
    from ripser import ripser
    from .metrics.topology.preservation import RIPSER_CONFIG
    grid = np.geomspace(.1,100,240) if eps_grid is None else np.asarray(eps_grid,dtype=float)
    if grid.ndim!=1 or len(grid)<2 or not np.isfinite(grid).all() or np.any(grid<=0) or np.any(np.diff(grid)<=0):
        raise ValueError('epsilon grid must be finite, positive and strictly increasing')
    diagrams = ripser(dist,**RIPSER_CONFIG)['dgms']
    curves = pd.DataFrame({'epsilon_sigma':grid})
    for dimension, diagram in enumerate(diagrams[:2]):
        curves[f'b{dimension}'] = np.sum((diagram[:,0,None]<=grid)&(diagram[:,1,None]>grid),axis=0)
    return curves, diagrams


def topology_scale_windows(diagrams, expected_betti, scale_bounds=(.1,100.)):
    """Exact connected scale intervals matching expected b0/b1, clipped to range.

    Bounds are left-inclusive/right-exclusive. Multiple disjoint windows are
    returned separately: spanning their envelope would invent a valid interval.
    For a Swiss roll, (1,0) also occurs after all topology has been filled in;
    this alone cannot establish preservation or absence of geometric shortcuts.
    """
    lower,upper=scale_bounds
    if not 0<lower<upper:raise ValueError('Require 0 < lower < upper')
    events=np.concatenate([diagram[np.isfinite(diagram)] for diagram in diagrams[:2]])
    edges=np.unique(np.r_[lower,events[(events>lower)&(events<upper)],upper])
    windows=[]
    for left,right in zip(edges[:-1],edges[1:]):
        counts=tuple(int(np.sum((diagram[:,0]<=left)&(diagram[:,1]>left))) for diagram in diagrams[:2])
        if counts==tuple(expected_betti):
            if windows and windows[-1][1]==left:windows[-1]=(windows[-1][0],float(right))
            else:windows.append((float(left),float(right)))
    return windows


def longest_betti_plateau(diagrams, dimension=0, scale_bounds=(.1,100.), nontrivial=False):
    """Longest interval of constant Betti count, by logarithmic scale width.

    Report linear and log widths explicitly. Optionally exclude b0=1, whose
    final trivial plateau otherwise often dominates. Range clipping matters.
    """
    lower,upper=scale_bounds
    diagram=diagrams[dimension]
    finite=diagram[np.isfinite(diagram)]
    edges=np.unique(np.r_[lower,finite[(finite>lower)&(finite<upper)],upper])
    intervals=[]
    for left,right in zip(edges[:-1],edges[1:]):
        height=int(np.sum((diagram[:,0]<=left)&(diagram[:,1]>left)))
        if nontrivial and height<=1:continue
        if intervals and intervals[-1][2]==height and intervals[-1][1]==left:
            intervals[-1]=(intervals[-1][0],float(right),height)
        else:intervals.append((float(left),float(right),height))
    if not intervals:return dict(lower_sigma=np.nan,upper_sigma=np.nan,width_sigma=0.,log_width=0.,height=np.nan)
    left,right,height=max(intervals,key=lambda item:np.log(item[1]/item[0]))
    return dict(lower_sigma=left,upper_sigma=right,width_sigma=right-left,log_width=float(np.log(right/left)),height=height)


def sigma_betti_over_epochs(run_folder, dataset=None, split='valid', config=None, eps_grid=None):
    """Posthoc Rips H0/H1 curves and exact topology windows from saved posteriors.

    No latent coordinates or missing parameters are reconstructed from sampled z.
    Known topology is a dataset assumption, used only in this posthoc analysis.
    A longest H1 bar need not track the same feature across snapshots. Sphere's
    (b0,b1) signature is partial; H2 is not computed. Beta need not move endpoints
    monotonically. Native VARIX clamps logvar >= 0.01, so tiny-sigma intuition
    from an unconstrained linear VAE is not assumed for these runs.
    """
    config=config or SigmaTopologyConfig(compute_h1=True)
    folder=Path(run_folder)
    expected={'circle':(1,1),'swiss_roll':(1,0),'sphere':(1,0)}.get(dataset)
    curves_all,summary_rows,window_rows=[],[],[]
    selected_ids=None;beta=_beta_by_epoch(folder)
    artifacts=folder/f'sigma_betti_{split}';artifacts.mkdir(exist_ok=True)
    for path in sorted((folder/'epoch_snapshots').glob(f'epoch_*_{split}.npz')):
        epoch=int(path.stem.split('_')[1]);context=dict(epoch=epoch,effective_beta=beta.get(epoch,np.nan),split=split)
        with np.load(path) as saved:
            lv_key='logvar' if 'logvar' in saved else 'sigmas'
            if not {'mu','sample_ids',lv_key}<=set(saved.files):
                summary_rows.append(dict(**context,status='missing_posterior'));continue
            ids=saved['sample_ids'].astype(str)
            if len(np.unique(ids))!=len(ids):raise ValueError('Snapshot IDs must be unique')
            if selected_ids is None:
                positions=np.arange(len(ids)) if len(ids)<=config.max_samples else np.sort(np.random.default_rng(31).choice(len(ids),config.max_samples,False))
                selected_ids=ids[positions]
            positions=pd.Index(ids).get_indexer(selected_ids)
            if (positions<0).any():raise ValueError('Snapshot cohort changed')
            mu,lv=saved['mu'][positions],saved[lv_key][positions]
        dist=sigma_unit_distances(mu,lv)
        curves,diagrams=betti_curves(dist,eps_grid)
        bounds=(float(curves.epsilon_sigma.min()),float(curves.epsilon_sigma.max()))
        curves_all.append(curves.assign(**context))
        row=dict(**context,status='ok',n_samples=len(mu),**sigma_unit_h0_summary(mu,lv))
        row.update(_main_h1_bar(diagrams[1]))
        row.update(posterior_sigma_min=float(np.exp(lv.min()/2)),
                   posterior_sigma_max=float(np.exp(lv.max()/2)))
        birth,death=row['h1_main_birth_sigma'],row['h1_main_death_sigma']
        for e in (1,2,3):
            row[f'h1_phase_at_eps{e}'] = ('no_finite_h1' if not np.isfinite(death) else
                'pre_birth' if e < birth else 'resolved' if e < death else 'post_death')
        for dimension in (0,1):
            np.save(artifacts/f'epoch_{epoch:05d}_h{dimension}.npy',diagrams[dimension])
            for nontrivial in ([False,True] if dimension==0 else [False]):
                prefix=f'b{dimension}_'+('nontrivial_' if nontrivial else '')+'plateau_'
                row.update({prefix+k:v for k,v in longest_betti_plateau(diagrams,dimension,bounds,nontrivial).items()})
        if expected is not None:
            windows=topology_scale_windows(diagrams,expected,bounds)
            for index,(left,right) in enumerate(windows):
                window_rows.append(dict(**context,window=index,lower_sigma=left,upper_sigma=right,width_sigma=right-left,
                                        log_width=np.log(right/left),left_inclusive=True,right_inclusive=False,
                                        **{f'contains_eps{e}':left<=e<right for e in (1,2,3)}))
            row['topology_window_count']=len(windows)
            for e in (1,2,3):row[f'topology_window_contains_eps{e}']=any(left<=e<right for left,right in windows)
            if windows:
                left,right=max(windows,key=lambda w:np.log(w[1]/w[0]))
                row.update(topology_window_lower_sigma=left,topology_window_upper_sigma=right,topology_window_width_sigma=right-left)
            else:row.update(topology_window_lower_sigma=np.nan,topology_window_upper_sigma=np.nan,topology_window_width_sigma=0.)
        summary_rows.append(row)
    summary=pd.DataFrame(summary_rows)
    summary.to_csv(folder/f'sigma_betti_{split}_summary.csv',index=False)
    pd.concat(curves_all,ignore_index=True).to_csv(folder/f'sigma_betti_{split}_curves.csv',index=False) if curves_all else pd.DataFrame().to_csv(folder/f'sigma_betti_{split}_curves.csv',index=False)
    pd.DataFrame(window_rows).to_csv(folder/f'sigma_topology_{split}_windows.csv',index=False)
    return summary


def plot_sigma_betti_curves(run_folder, split='valid'):
    """Plot the full epsilon grid per observed epoch without interpolating gaps."""
    import matplotlib.pyplot as plt
    folder=Path(run_folder)
    path=folder/f'sigma_betti_{split}_curves.csv'
    try:curves=pd.read_csv(path)
    except pd.errors.EmptyDataError:return None
    if curves.empty:return None
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for dimension,ax in enumerate(axes):
        for epoch,group in curves.groupby('epoch'):
            ax.step(group.epsilon_sigma,group[f'b{dimension}'],where='post',alpha=.5,
                    label=f'Epoch {epoch}')
        ax.set(xscale='log',xlabel='ε (σ-Einheiten)',title=f'b{dimension}(ε)')
    axes[0].set_ylabel('Betti number')
    if curves.epoch.nunique()<=10:axes[-1].legend(fontsize=7)
    fig.tight_layout();path=folder/f'sigma_betti_{split}_curves.png'
    fig.savefig(path,dpi=160);plt.close(fig)
    return path
