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
    In core metrics included: no; optional complementary analysis.
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
    from beta_vae_experiments.metrics.information_theoretic import compute_conditional_kl_matrix
    from beta_vae_experiments.metrics.latent_usage import PosteriorCollapseConfig
    policy = PosteriorCollapseConfig()
    kl = compute_conditional_kl_matrix(mu, logvar)
    return int(np.sum(np.mean(kl < policy.primary_epsilon, axis=0) >= 1-policy.delta))


@dataclass(frozen=True)
class SigmaTopologyConfig:
    enabled: bool = False
    eps: tuple[float,...] = (1.,2.,3.)
    max_samples: int = 1000
    compute_h1: bool = False

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('enabled must be boolean')
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
    from beta_vae_experiments.metrics.topology.preservation import RIPSER_CONFIG
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
    if not config.enabled:
        return pd.DataFrame()
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


@dataclass(frozen=True)
class H0AnnealingConfig:
    """Validation-only staircase; H0 overlap limits are not collapse thresholds."""
    mode: str = 'h0'
    rule: str = 'plateau'
    eps: float = 2.
    frac_max: float = .10
    patience: int = 5
    gap_margin: float = 4.
    plateau_gap_factor: float = 4.
    plateau_min_merged: float = .5
    warmup_epochs: int = 100
    w_min: float = 1e-4
    growth: float = 1.25
    step_epochs: int = 20
    ema_epochs: int = 3
    n_subsample: int = 800
    seed: int = 31
    w_max: float = 1.

    def __post_init__(self):
        if self.mode != 'h0' or self.rule not in ('keep_fine', 'plateau'):
            raise ValueError('Expected mode=h0 and rule=keep_fine or plateau')
        positive = (self.eps, self.gap_margin, self.plateau_gap_factor,
                    self.w_min, self.growth, self.w_max)
        if any(not np.isfinite(x) or x <= 0 for x in positive):
            raise ValueError('Thresholds and weights must be positive and finite')
        if not 0 <= self.frac_max <= 1 or not 0 <= self.plateau_min_merged <= 1:
            raise ValueError('Merge fractions must lie in [0,1]')
        if self.growth <= 1 or self.w_min > self.w_max:
            raise ValueError('Require growth>1 and w_min<=w_max')
        for name in ('patience', 'step_epochs', 'ema_epochs', 'n_subsample'):
            if type(getattr(self,name)) is not int or getattr(self,name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if (type(self.warmup_epochs) is not int or self.warmup_epochs < 0
                or self.ema_epochs > self.step_epochs or self.n_subsample < 2):
            raise ValueError('Invalid warmup, averaging window or subsample size')


@dataclass(frozen=True)
class H0Stage:
    w: float
    previous_w: float = 0.
    stage: int = 1
    state: str = 'ramp'
    history: tuple = ()
    reason: str = ''


def advance_h0_stage(stage, summary, config):
    """Pure end-of-stage rule, using the mean of the stage's last measurements.

    Plateau detection requires stable nontrivial component counts, a bounded
    maximum gap, and sufficient completed merges. The gap emergency and
    keep_fine roll back one weight; an accepted plateau retains its weight.
    """
    if stage.state == 'hold':
        return stage
    tag = _eps_tag(config.eps)
    count = float(summary['h0_components_at_eps'+tag])
    fraction = float(summary['h0_frac_below_eps'+tag])
    gap = float(summary['h0_max_gap_sigma'])
    if not np.isfinite([count, fraction, gap]).all():
        raise ValueError('Stage decisions require finite H0 measurements')
    history = stage.history + ((count, fraction, gap),)
    if config.rule == 'keep_fine' and fraction > config.frac_max:
        return replace(stage, w=stage.previous_w, state='hold', history=history, reason='fine_merge_limit')
    if config.rule == 'plateau':
        if gap < config.gap_margin:
            return replace(stage, w=stage.previous_w, state='hold', history=history, reason='gap_emergency')
        window = history[-config.patience:]
        stable = (len(history) >= config.patience and count > 1
                  and all(item[0] == count for item in window))
        enough_merges = all(item[1] >= config.plateau_min_merged for item in window)
        if stable and enough_merges and gap < config.plateau_gap_factor*config.gap_margin:
            return replace(stage, state='hold', history=history, reason='stable_plateau')
    if stage.w >= config.w_max:
        return replace(stage, state='hold', history=history, reason='weight_cap')
    return H0Stage(w=min(stage.w*config.growth, config.w_max), previous_w=stage.w,
                   stage=stage.stage+1, history=history)


def controller_tag(settings):
    validated = asdict(H0AnnealingConfig(**settings))
    digest = hashlib.sha256(json.dumps(validated,sort_keys=True).encode()).hexdigest()[:10]
    return 'h0-'+validated['rule']+'-'+digest


def make_h0_loss(settings, config, run_folder, n_valid):
    """Observe ordered validation batches through the native VarixLoss hook.

    Only validation posterior parameters enter decisions. Its fixed cohort is
    selected once by row position; labels, training and test posteriors are unused.
    A stage uses the arithmetic mean of its last ema_epochs measurements.
    """
    import torch
    from autoencodix.utils._losses import VarixLoss
    policy = H0AnnealingConfig(**settings)
    if config.default_vae_loss != 'kl' or config.loss_reduction != 'mean':
        raise ValueError('H0 annealing requires KL and mean reduction')
    if config.anneal_function == 'no-annealing':
        raise ValueError('no-annealing bypasses custom schedulers; use 3phase-linear')
    if n_valid < 2:
        raise ValueError('H0 annealing requires at least two validation patients')
    selected = np.sort(np.random.default_rng(policy.seed).choice(n_valid,min(n_valid,policy.n_subsample),False))
    folder = Path(run_folder)
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'controller_config.json').write_text(json.dumps(dict(
        settings=asdict(policy),n_validation=n_valid,validation_positions=selected.tolist(),
        cohort='fixed positions in the native ordered validation loader; no labels',
    ),indent=2)+'\n')

    class H0Scheduler:
        def __init__(self):
            self.w = 0. if policy.warmup_epochs else policy.w_min

        def get_weight(self, *, epoch_current, total_epoch, func):
            return self.w

    class H0VarixLoss(VarixLoss):
        def __init__(self, config, annealing_scheduler=None):
            if annealing_scheduler is not None:
                raise ValueError('Cannot combine annealers')
            self.h0_scheduler = H0Scheduler()
            super().__init__(config,annealing_scheduler=self.h0_scheduler)
            self.h0_stage = H0Stage(w=self.h0_scheduler.w, stage=0 if policy.warmup_epochs else 1,
                                    state='warmup' if policy.warmup_epochs else 'ramp')
            self.h0_epoch = None
            self.h0_offset = 0
            self.h0_means, self.h0_logvars = [], []
            self.h0_stage_measurements = []
            self.h0_logged = set()
            self.h0_rows = []

        def finish_validation(self):
            if self.h0_epoch is None or self.h0_epoch in self.h0_logged:
                return
            if self.h0_offset != n_valid:
                raise ValueError(f'Incomplete ordered validation cohort: {self.h0_offset}/{n_valid}')
            mu, lv = np.concatenate(self.h0_means), np.concatenate(self.h0_logvars)
            thresholds = tuple(sorted({1.,2.,3.,policy.eps}))
            summary = sigma_unit_h0_summary(mu,lv,thresholds)
            row = dict(epoch=self.h0_epoch+1,stage=self.h0_stage.stage,w=self.h0_scheduler.w,
                       effective_beta=config.beta*self.h0_scheduler.w,
                       beta_over_d=config.beta*self.h0_scheduler.w/config.latent_dim,
                       state=self.h0_stage.state,reason=self.h0_stage.reason,
                       seed=config.global_seed,latent_dim=config.latent_dim,**summary)
            self.h0_rows.append(row)
            pd.DataFrame(self.h0_rows).to_csv(folder/'controller_log.csv',index=False)
            pd.DataFrame([dict(epoch=r['epoch'],beta_over_d=r['beta_over_d'],
                               beta_config=r['effective_beta']) for r in self.h0_rows]).to_csv(folder/'beta_schedule.csv',index=False)
            self.h0_logged.add(self.h0_epoch)
            if self.h0_stage.state == 'ramp':
                self.h0_stage_measurements.append(summary)
                if len(self.h0_stage_measurements) == policy.step_epochs:
                    tail = self.h0_stage_measurements[-policy.ema_epochs:]
                    averaged = {key:float(np.mean([item[key] for item in tail])) for key in summary}
                    self.h0_stage = advance_h0_stage(self.h0_stage,averaged,policy)
                    self.h0_stage_measurements = []
            self.h0_offset = 0
            self.h0_means, self.h0_logvars = [], []

        def forward(self, model_output, targets, epoch=None, **kwargs):
            if epoch is not None and torch.is_grad_enabled() and epoch != self.h0_epoch:
                self.finish_validation()
                if self.h0_stage.state == 'warmup' and epoch >= policy.warmup_epochs:
                    self.h0_stage = H0Stage(w=policy.w_min)
                self.h0_scheduler.w = self.h0_stage.w
                self.h0_epoch = int(epoch)
            if epoch is not None and not torch.is_grad_enabled():
                count = len(model_output.latent_mean)
                local = selected[(selected>=self.h0_offset)&(selected<self.h0_offset+count)]-self.h0_offset
                if len(local):
                    self.h0_means.append(model_output.latent_mean[local].detach().cpu().numpy().astype(np.float64))
                    # Native latent_logvar/_sigma_buffer is log variance, NOT sigma.
                    self.h0_logvars.append(model_output.latent_logvar[local].detach().cpu().numpy().astype(np.float64))
                self.h0_offset += count
            result = super().forward(model_output=model_output,targets=targets,epoch=epoch,**kwargs)
            # Varix.run() reinitializes its trainer/loss for final prediction.
            # Flush the last validation before that reset, since no next epoch exists.
            if (epoch == config.epochs-1 and not torch.is_grad_enabled()
                    and self.h0_offset == n_valid):
                self.finish_validation()
            return result

        def finalize(self):
            self.finish_validation()

    return H0VarixLoss


def evaluate_sigma_snapshots(run_folder, varix=None):
    """Post-training label-based validation on mu and recorded z; never controls beta.

    Linear/logistic and 15-neighbor probes share train-defined snapshot_tasks.
    Classification uses macro OvO AUC; regression uses R2. Native train arrays
    were collected during updates, so these are checkpoint diagnostics rather
    than fresh encodings with a common frozen model. Missing arrays stay missing.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.metrics import roc_auc_score, r2_score
    from beta_vae_experiments.metrics.evaluation.downstream import snapshot_tasks
    from beta_vae_experiments.metrics.information_theoretic import compute_conditional_kl_matrix
    folder = Path(run_folder)/'epoch_snapshots'
    rows, diagnostics, dimensions = [], [], []
    beta = _beta_by_epoch(run_folder)
    for train_path in sorted(folder.glob('epoch_*_train.npz')):
        epoch = int(train_path.stem.split('_')[1])
        valid_path = folder/train_path.name.replace('_train','_valid')
        if not valid_path.exists():
            continue
        with np.load(train_path) as train, np.load(valid_path) as valid:
            arrays = {'train':dict(train),'valid':dict(valid)}
        labels = {}
        for split, values in arrays.items():
            label_path = folder/f'labels_{split}.parquet'
            if not label_path.exists() or 'sample_ids' not in values:
                break
            frame = pd.read_parquet(label_path)
            frame.index = frame.index.astype(str)
            labels[split] = frame.loc[values['sample_ids'].astype(str)].copy()
        valid = arrays['valid']
        lv = valid.get('logvar',valid.get('sigmas'))
        if 'mu' in valid and lv is not None:
            kl = compute_conditional_kl_matrix(valid['mu'],lv)
            diagnostics.append(dict(epoch=epoch,effective_beta=beta.get(epoch,np.nan),
                                    rate=float(kl.sum(1).mean()),n_collapsed_dims_kl=_collapsed_dimensions(valid['mu'],lv)))
            for dimension in range(kl.shape[1]):
                dimensions.append(dict(epoch=epoch,dimension=dimension,kl_mean=kl[:,dimension].mean(),
                                       fraction_below_0p01=np.mean(kl[:,dimension]<.01),
                                       is_collapsed=np.mean(kl[:,dimension]<.01)>=.99))
        if varix is not None:
            from beta_vae_experiments.data_pipeline import to_numpy
            reconstruction = to_numpy(varix.result.reconstructions.get(epoch=epoch-1,split='valid'))
            if reconstruction is not None and len(reconstruction):
                actual = to_numpy(varix.result.datasets.valid.data)
                diagnostics.append(dict(epoch=epoch,effective_beta=beta.get(epoch,np.nan),
                                        reconstruction_r2=r2_score(actual,reconstruction)))
        if set(labels) != {'train','valid'}:
            continue
        for task in snapshot_tasks(labels):
            for representation in ('mu','z'):
                if any(representation not in values for values in arrays.values()):
                    continue
                keep = task['keep']
                x = {s:arrays[s][representation][keep[s]] for s in arrays}
                y = {s:task['y'][s].to_numpy()[keep[s]] for s in arrays}
                for method in ('linear','knn15'):
                    row = dict(epoch=epoch,effective_beta=beta.get(epoch,np.nan),task=task['name'],
                               representation=representation,method=method,n_train=len(y['train']),n_valid=len(y['valid']))
                    try:
                        if len(y['train'])<15 or len(y['valid'])<2:
                            raise ValueError('Insufficient cohort for shared 15-neighbor comparison')
                        if task['regression']:
                            model = Ridge() if method=='linear' else KNeighborsRegressor(15)
                            model.fit(x['train'],y['train'].astype(float))
                            score = r2_score(y['valid'].astype(float),model.predict(x['valid']))
                            score_name='r2'
                        else:
                            model = LogisticRegression(max_iter=1000) if method=='linear' else KNeighborsClassifier(15)
                            model.fit(x['train'],y['train'])
                            probabilities = model.predict_proba(x['valid'])
                            if set(y['valid']) != set(model.classes_):
                                raise ValueError('Validation cohort does not cover every training class')
                            score = (roc_auc_score(y['valid'],probabilities[:,1]) if len(model.classes_)==2 else
                                     roc_auc_score(y['valid'],probabilities,labels=model.classes_,multi_class='ovo',average='macro'))
                            score_name='auc_ovo'
                        rows.append(dict(**row,score=score,score_name=score_name,status='ok',warning=None))
                    except ValueError as error:
                        rows.append(dict(**row,score=np.nan,status='unavailable',warning=str(error)))
    result = pd.DataFrame(rows)
    result.to_csv(Path(run_folder)/'sigma_downstream.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(Path(run_folder)/'sigma_validation_diagnostics.csv',index=False)
    pd.DataFrame(dimensions).to_csv(Path(run_folder)/'sigma_kl_dimensions.csv',index=False)
    return result


def plot_sigma_training_curves(run_folder, latent_dim):
    """H0 and downstream curves versus realized beta/d; chronology is retained.

    Drop markers denote a 5% relative decrease from the first available AUC.
    Vertical plateau markers indicate accepted controller plateaus, not proof
    of task preservation. Repeated beta values can have distinct training ages.
    """
    import matplotlib.pyplot as plt
    folder=Path(run_folder)
    betti_path=folder/'sigma_betti_valid_summary.csv'
    topology=pd.read_csv(betti_path if betti_path.exists() else folder/'sigma_topology_valid_epochs.csv')
    path=folder/'sigma_downstream.csv'
    try: downstream=pd.read_csv(path)
    except pd.errors.EmptyDataError: downstream=pd.DataFrame()
    tasks=[] if downstream.empty else downstream.task.unique().tolist()
    fig,axes=plt.subplots(3+len(tasks),1,figsize=(10,3*(3+len(tasks))),sharex=True,squeeze=False)
    axes=axes[:,0]
    for ax,key in zip(axes,('nn_median_sigma','h0_max_gap_sigma','h1_main_persistence_sigma')):
        if key in topology:
            ax.plot(topology.effective_beta/latent_dim,topology[key],'.-')
        ax.set_ylabel(key)
    for ax,task in zip(axes[3:],tasks):
        for (representation,method), group in downstream[downstream.task.eq(task)].groupby(['representation','method']):
            group=group[group.status.eq('ok')].sort_values('epoch')
            if group.empty:continue
            x=group.effective_beta/latent_dim
            ax.plot(x,group.score,'.-',label=f'{representation}: {method}')
            if group.iloc[0].get('score_name')=='auc_ovo':
                drops=group[group.score<.95*group.iloc[0].score]
                if not drops.empty:ax.axvline(drops.iloc[0].effective_beta/latent_dim,color='red',alpha=.3)
        ax.set_ylabel(task);ax.legend(fontsize=7)
    log_path=folder/'controller_log.csv'
    if log_path.exists():
        log=pd.read_csv(log_path)
        for value in log.loc[log.reason.eq('stable_plateau'),'beta_over_d'].unique():
            for ax in axes:ax.axvline(value,color='green',linestyle='--',alpha=.5)
    for ax in axes:ax.set_xscale('symlog',linthresh=1e-8)
    axes[-1].set_xlabel('Realized beta / latent dimension (zero shown on symlog axis)')
    fig.tight_layout();fig.savefig(folder/'sigma_training_curves.png',dpi=160);plt.close(fig)


def analyze_sigma_run(varix, run_folder, dataset=None):
    """Posthoc-only labels and plots, independently of the validation controller."""
    sigma_topology_over_epochs(run_folder)
    sigma_betti_over_epochs(run_folder,dataset=dataset)
    plot_sigma_betti_curves(run_folder)
    evaluate_sigma_snapshots(run_folder,varix)
    plot_sigma_training_curves(run_folder,varix.config.latent_dim)


def apply_fixed_beta_baseline(settings, controller_log):
    """Match one completed controller run's last *realized* beta, seed and dimension.

    Invoke once per controller log; do not silently broadcast an endpoint from
    one seed onto other seeds. The source log must represent a completed run.
    """
    import copy
    settings=copy.deepcopy(settings)
    table=pd.read_csv(controller_log)
    required={'epoch','effective_beta','seed','latent_dim'}
    if not required <= set(table) or table.empty or table.epoch.duplicated().any():
        raise ValueError('Expected a nonempty controller log with unique epochs, seed and dimension')
    if table.seed.nunique()!=1 or table.latent_dim.nunique()!=1:
        raise ValueError('Controller log must belong to exactly one seed and dimension')
    last=table.sort_values('epoch').iloc[-1]
    if int(last.epoch)!=int(settings['training']['epochs']):
        raise ValueError('Controller log must cover the baseline training duration')
    training=settings['training']
    training.pop('beta_controller',None);training.pop('beta_schedule',None);training.pop('betas',None)
    training.pop('require_controller_endpoint',None)
    training.update(anneal_function='no-annealing',beta_over_d=[float(last.effective_beta/last.latent_dim)],
                    seeds=[int(last.seed)],latent_dims=[int(last.latent_dim)])
    settings['fixed_beta_controller_source']=str(Path(controller_log).resolve())
    endpoint_tag=hashlib.sha256(Path(controller_log).read_bytes()).hexdigest()[:12]
    settings['output_dir']=Path(settings['output_dir'])/('endpoint-'+endpoint_tag)
    # Split identity is checked against the source experiment, not inferred from seed.
    source=Path(controller_log).resolve().parents[2]/'experiment_config.json'
    if not source.exists():raise ValueError('Source experiment configuration is required for cohort matching')
    original=json.loads(source.read_text())
    if original['dataset']!=settings['dataset']:
        raise ValueError('Fixed endpoint baseline must use the same dataset options and split')
    return settings


def betti_curves(dist, eps_grid=None):
    """Sample b0/b1 over scale; additionally retain exact diagram breakpoints.

    Default log grid [0.1,100] is an analysis range, not a universal scale.
    Standard Rips convention: birth <= epsilon < death (unlike legacy strict
    merge summaries). Finite point sampling can create short spurious H1 bars.
    """
    from ripser import ripser
    from beta_vae_experiments.metrics.topology.preservation import RIPSER_CONFIG
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


def analyze_complete_circle_runs(results_root, output_dir):
    """Compare complete archived Circle runs, without training or filling gaps.

    Require every configured Train/Validation snapshot and actual mu/logvar/z/IDs.
    Circle sin/cos targets are continuous: downstream quality is R2, not AUC.
    Curves reuse native recorded z (one draw), not a new Monte Carlo estimate.
    Train posteriors were captured during batch updates, as in native snapshots.
    """
    import matplotlib.pyplot as plt
    results_root, output_dir = Path(results_root), Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    manifest, excluded, topology_frames, probe_frames, rate_frames = [], [], [], [], []
    for experiment in sorted(results_root.glob('*circle*')):
        config_path, overview_path = experiment/'experiment_config.json', experiment/'experiment_metrics.csv'
        if not config_path.exists() or not overview_path.exists():
            continue
        settings = json.loads(config_path.read_text())
        interval = settings['training'].get('snapshots',{}).get('every',50)
        epochs = int(settings['training']['epochs'])
        expected_epochs = set(range(interval,epochs+1,interval)) | {epochs}
        overview = pd.read_csv(overview_path)
        for _, run in overview[overview.status.eq('ok')].iterrows():
            folder = experiment/'run_analysis'/run.run_name
            problem = None
            for split in ('train','valid'):
                for epoch in sorted(expected_epochs):
                    path = folder/'epoch_snapshots'/f'epoch_{epoch:05d}_{split}.npz'
                    if not path.exists():
                        problem = f'Missing {split} snapshot at epoch {epoch}'
                        break
                    with np.load(path) as saved:
                        if not {'mu','logvar','z','sample_ids'} <= set(saved.files):
                            problem = f'Missing actual posterior arrays: {split}, epoch {epoch}'
                            break
                if problem:
                    break
            context = dict(run_name=run.run_name,experiment=experiment.name,
                           split_seed=settings['dataset']['split_seed'],seed=int(run.seed),
                           target_beta=float(run.beta),latent_dim=int(run.latent_dim))
            if problem:
                excluded.append(dict(**context,reason=problem))
                continue
            print(f'Analyzing complete Circle run: split {context["split_seed"]}, seed {context["seed"]}',flush=True)
            manifest.append(dict(**context,epochs=epochs,n_snapshots=len(expected_epochs),run_folder=str(folder.resolve())))
            topology_path = folder/'sigma_betti_valid_summary.csv'
            if not topology_path.exists():
                sigma_betti_over_epochs(folder,dataset='circle')
            topology = pd.read_csv(topology_path)
            if set(topology.epoch) != expected_epochs or not topology.status.eq('ok').all():
                raise ValueError(f'Incomplete topology summary for {run.run_name}')
            topology_frames.append(topology.assign(**context))
            probes = evaluate_sigma_snapshots(folder)
            probe_frames.append(probes.assign(**context))
            rates = pd.read_csv(folder/'sigma_validation_diagnostics.csv')
            rate_frames.append(rates.assign(**context))
            plot_sigma_betti_curves(folder)
            plot_sigma_training_curves(folder,int(run.latent_dim))
            if int(run.latent_dim)==2:
                _plot_circle_snapshot_coordinates(folder,output_dir,context,sorted(expected_epochs))
    pd.DataFrame(manifest).to_csv(output_dir/'included_runs.csv',index=False)
    pd.DataFrame(excluded).to_csv(output_dir/'excluded_runs.csv',index=False)
    if not manifest:
        raise ValueError('No complete Circle runs found')
    topology = pd.concat(topology_frames,ignore_index=True)
    probes = pd.concat(probe_frames,ignore_index=True)
    rates = pd.concat(rate_frames,ignore_index=True)
    topology.to_csv(output_dir/'circle_topology_epochs.csv',index=False)
    probes.to_csv(output_dir/'circle_downstream_epochs.csv',index=False)
    rates.to_csv(output_dir/'circle_rate_epochs.csv',index=False)
    fig,axes = plt.subplots(3,2,figsize=(13,12),sharex=True)
    for _, group in topology.groupby('run_name'):
        for ax, key in zip(axes.flat[:4],('h1_main_birth_sigma','h1_main_death_sigma','h1_main_persistence_sigma','nn_median_sigma')):
            ax.plot(group.epoch,group[key],'.-',alpha=.55)
            ax.set_ylabel(key)
    for ax in axes.flat[:2]:
        ax.axhline(2,color='red',linestyle='--',label='2 sigma')
        ax.set_yscale('symlog',linthresh=.01)
    for _,group in rates.groupby('run_name'):
        axes[2,0].plot(group.epoch,group.rate,alpha=.55)
    axes[2,0].set_ylabel('Rate (nats per patient)')
    successful = probes[probes.status.eq('ok')]
    for (representation,method),group in successful.groupby(['representation','method']):
        # Average sin/cos only for the overview; individual target scores stay in CSV.
        per_run = group.groupby(['run_name','epoch']).score.mean().reset_index()
        median = per_run.groupby('epoch').score.median()
        axes[2,1].plot(median.index,median,label=f'{representation}, {method}')
    axes[2,1].set_ylabel('Median R2 (mean of sin/cos per run)')
    axes[2,1].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Epoch')
    fig.suptitle('Complete Circle runs: archived validation posteriors, no new training')
    fig.tight_layout();fig.savefig(output_dir/'circle_complete_runs.png',dpi=170);plt.close(fig)
    final = topology[topology.epoch.eq(epochs)]
    phases = topology.h1_phase_at_eps2.value_counts().to_dict()
    report = dict(n_runs=len(manifest),n_excluded=len(excluded),n_snapshots=len(topology),
                  h1_phases_at_eps2=phases,finite_h1_snapshots=int(topology.h1_main_death_sigma.notna().sum()),
                  resolved_at_eps2=int(topology.h1_phase_at_eps2.eq('resolved').sum()),
                  final_rate_min=float(rates.loc[rates.epoch.eq(epochs),'rate'].min()),
                  final_rate_max=float(rates.loc[rates.epoch.eq(epochs),'rate'].max()),
                  final_collapsed_dims=final.n_collapsed_dims_kl.value_counts().to_dict(),
                  limitation='First snapshot at epoch 50; transitions before it are unobserved. No claim of causal annealer performance.')
    (output_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def _plot_circle_snapshot_coordinates(run_folder, output_dir, context, epochs):
    """Compare stored mu and z colored by angle, with exact ID alignment."""
    import matplotlib.pyplot as plt
    folder = Path(run_folder)/'epoch_snapshots'
    labels = pd.read_parquet(folder/'labels_valid.parquet')
    labels.index = labels.index.astype(str)
    if 'ground_truth_angle' not in labels:
        return
    selected = [epochs[0],epochs[len(epochs)//2-1],epochs[-1]]
    fig,axes = plt.subplots(2,3,figsize=(12,7))
    for column,epoch in enumerate(selected):
        with np.load(folder/f'epoch_{epoch:05d}_valid.npz') as saved:
            angle = labels.loc[saved['sample_ids'].astype(str),'ground_truth_angle'].to_numpy()
            for row,key in enumerate(('mu','z')):
                values = saved[key]
                axes[row,column].scatter(values[:,0],values[:,1],c=angle,cmap='twilight',vmin=0,vmax=2*np.pi,s=7,alpha=.65)
                axes[row,column].set(title=f'{key}, epoch {epoch}',xlabel='Latent coordinate 1',ylabel='Latent coordinate 2')
                axes[row,column].set_aspect('equal',adjustable='datalim')
    fig.suptitle(f'Circle: split {context["split_seed"]}, seed {context["seed"]}; color = true angle')
    fig.tight_layout()
    fig.savefig(Path(output_dir)/f'circle_coordinates_split{context["split_seed"]}_seed{context["seed"]}.png',dpi=160)
    plt.close(fig)
