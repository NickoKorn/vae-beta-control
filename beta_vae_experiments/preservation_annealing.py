"""Validation-based beta staircases through AUTOENCODIX's native VarixLoss hook.

Trustworthiness, MRRE false, LCMC and Cohesiveness are label-free X-to-mu signals.
Their association with Cancer or Sex is a hypothesis evaluated after training.
Targets are explicit absolute scores or fractions of a fixed warm-up score.
Neither interpretation is a percentage of retained downstream information.
"""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreservationAnnealingConfig:
    mode: str = 'preservation'
    metric: str = 'trustworthiness'
    target: float = .75
    target_mode: str = 'absolute'
    rule: str = 'minimum'
    tolerance: float = 0.01
    max_invalid_measurements: int = 3
    warmup_epochs: int = 100
    step_epochs: int = 20
    w_min: float = 1e-4
    growth: float = 1.25
    n_subsample: int = 800
    k: int = 20
    seed: int = 31

    def __post_init__(self):
        if self.mode != 'preservation' or self.metric not in ('trustworthiness','cohesiveness','mrre_false','lcmc'):
            raise ValueError('Expected preservation mode and trustworthiness, mrre_false, lcmc or cohesiveness')
        if self.target_mode not in ('absolute', 'relative_warmup'):
            raise ValueError('target_mode must be absolute or relative_warmup')
        if self.rule not in ('minimum', 'target'):
            raise ValueError('rule must be minimum or target')
        if not np.isfinite(self.tolerance) or not 0 < self.tolerance < 1:
            raise ValueError('tolerance must lie in (0,1)')
        if not np.isfinite(self.target) or not 0 < self.target <= 1:
            raise ValueError('target must be a score in (0,1]')
        for name in ('warmup_epochs','step_epochs','n_subsample','k','seed','max_invalid_measurements'):
            value = getattr(self,name)
            minimum = 0 if name=='seed' else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f'Invalid integer {name}')
        if self.seed >= 2**32:
            raise ValueError('seed must fit in an unsigned 32-bit integer')
        if self.n_subsample <= 2*self.k:
            raise ValueError('n_subsample must exceed 2*k')
        if not np.isfinite(self.w_min) or not 0 < self.w_min <= 1:
            raise ValueError('Require 0 < w_min <= 1')
        if not np.isfinite(self.growth) or self.growth <= 1:
            raise ValueError('growth must exceed 1')


@dataclass(frozen=True)
class PreservationStage:
    w: float = 0.
    previous_w: float = 0.
    state: str = 'warmup'
    reason: str = ''
    baseline: float | None = None
    invalid_measurements: int = 0
    last_direction: int = 0
    search_step: float | None = None


def preservation_target(stage, config):
    """Raw metric threshold; relative targets are fixed after warm-up calibration."""
    if config.target_mode == 'absolute':
        return config.target
    if stage.baseline is not None and stage.baseline <= 0:
        raise ValueError('Relative preservation requires a positive warm-up baseline')
    return np.nan if stage.baseline is None else config.target * stage.baseline


def advance_preservation_stage(stage, score, config):
    """Pure staircase rule with explicit calibration and bounded recovery.

    An unreachable absolute target fails rather than quietly training at beta=0.
    Invalid observations back off to the previous weight; repeated invalidity
    fails the run. Reducing beta does not guarantee recovery of lost information.
    """
    if not np.isfinite(score):
        invalid = stage.invalid_measurements + 1
        if invalid >= config.max_invalid_measurements:
            raise ValueError(f'{config.metric} unavailable for {invalid} consecutive measurements; '
                             'refusing to train without a valid preservation signal')
        # Target tracking also decreases beta, so the previous weight may be
        # higher. An invalid observation must never increase compression.
        rollback = min(stage.previous_w, stage.w/config.growth) if stage.w > 0 else 0.
        previous = 0. if rollback <= config.w_min * (1+1e-12) else rollback/config.growth
        return replace(stage, w=rollback, previous_w=previous, invalid_measurements=invalid,
                       state='recovering' if stage.w > 0 else 'waiting_for_metric', reason='metric_unavailable')
    stage = replace(stage, baseline=score if stage.baseline is None else stage.baseline,
                    invalid_measurements=0)
    threshold = preservation_target(stage, config)
    lower = threshold - config.tolerance if config.rule == 'target' else threshold
    if (stage.w == 0 and stage.state in ('warmup', 'waiting_for_metric')
            and config.target_mode == 'absolute' and score < lower):
        raise ValueError(f'Warm-up {config.metric}={score:.6g} is below absolute target={threshold:.6g}; '
                         'calibrate the target or explicitly select relative_warmup')
    if config.rule == 'target':
        return seek_preservation_target(stage, score, threshold, config)
    if stage.state == 'hold':
        return stage
    if score <= threshold:
        rollback = stage.previous_w
        previous = 0. if rollback <= config.w_min*(1+1e-12) else rollback/config.growth
        return replace(stage, w=rollback, previous_w=previous, state='hold', reason='target_not_met')
    if stage.state == 'recovering':
        return replace(stage, state='hold', reason='recovered_after_invalid')
    if stage.w == 0:
        return replace(stage, w=config.w_min, previous_w=0., state='ramp', reason='')
    if stage.w >= 1:
        return replace(stage, state='hold', reason='weight_cap')
    return replace(stage, w=min(1.,stage.w*config.growth), previous_w=stage.w, state='ramp', reason='')


def seek_preservation_target(stage, score, target, config):
    """Bidirectional beta feedback with a deadband, not a hard model constraint.

    Scores above target increase beta; scores below decrease beta. Reversals
    halve the logarithmic step (minimum 0.1%) to reduce overshoot. A held weight
    can change again if the score leaves the band. Nonmonotone training dynamics
    and beta bounds can prevent convergence; no root-finding guarantee is made.
    """
    if abs(score-target) <= config.tolerance + 1e-12:
        return replace(stage, state='hold', reason='target_band')
    direction = 1 if score > target else -1
    step = stage.search_step if stage.search_step is not None else float(np.log(config.growth))
    if stage.last_direction and direction != stage.last_direction:
        step = max(step/2, float(np.log1p(.001)))
    weight = config.w_min if stage.w == 0 and direction > 0 else stage.w*np.exp(direction*step)
    weight = float(np.clip(weight, 0., 1.))
    if direction < 0 and weight < config.w_min:
        weight = 0.
    bounded = weight == stage.w
    reason = ('weight_cap' if direction > 0 else 'zero_weight_bound') if bounded else (
        'above_target' if direction > 0 else 'below_target')
    return replace(stage, w=weight, previous_w=stage.w, state='hold' if bounded else 'seeking',
                   reason=reason, last_direction=direction, search_step=step)


def _self_first_neighbours(values, k):
    """Neighbour lists and rankings in which every point is its own rank 0.

    ZADU drops the first sorted column as "self". With exact duplicates (Varix
    sets mu < 1e-7 to 0, so many patients share the origin) another identical
    point can come first: self stays in the list and a true neighbour is lost.
    Here the diagonal is pushed below all distances, as sklearn excludes self by
    index. Remaining ties are ordered by sample index (stable sort); that order
    is a convention, not information.
    """
    from zadu.measures.utils.pairwise_dist import pairwise_distance_matrix
    distances = np.array(pairwise_distance_matrix(values, 'euclidean'), dtype=np.float64)
    np.fill_diagonal(distances, -1.)
    order = np.argsort(distances, axis=1, kind='stable')
    return order[:, 1:k+1], np.argsort(order, axis=1, kind='stable')


def measure_preservation(x, mu, config):
    """Official sklearn trustworthiness or a selected ZADU score; no custom formula.

    sklearn excludes self by index, including for tied/duplicate embeddings.
    ZADU MRRE and LCMC receive self-first neighbour lists through their official
    knn_ranking_info / knn_info arguments, so duplicate mu (patients at the Varix
    zero clamp) no longer bias them; the formulas stay ZADU's. Without this,
    duplicates made ZADU MRRE false optimistic by up to 0.06 after collapse.
    Distance ties still make the neighbor ordering convention-dependent.
    Constant embeddings are withheld. ZADU SNC keeps the existing stricter
    float32-duplicate guard and uses floor(sqrt(n)), not the supplied k.
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html
    https://github.com/hj-n/zadu
    """
    x, mu = np.asarray(x), np.asarray(mu)
    if not np.isfinite(x).all() or not np.isfinite(mu).all():
        raise ValueError('Nonfinite preservation coordinates')
    if np.all(mu == mu[0]):
        raise ValueError('Constant/duplicate embedding: preservation signal unavailable')
    if config.metric == 'trustworthiness':
        from sklearn.manifold import trustworthiness
        return float(trustworthiness(x, mu, n_neighbors=config.k))
    if config.metric in ('mrre_false', 'lcmc'):
        from zadu.measures import local_continuity_meta_criteria, mean_relative_rank_error
        x_neighbours, x_ranking = _self_first_neighbours(x, config.k)
        mu_neighbours, mu_ranking = _self_first_neighbours(mu, config.k)
        if config.metric == 'mrre_false':
            result = mean_relative_rank_error.measure(
                x, mu, k=config.k, knn_ranking_info=(x_neighbours, x_ranking, mu_neighbours, mu_ranking))
        else:
            result = local_continuity_meta_criteria.measure(x, mu, k=config.k, knn_info=(x_neighbours, mu_neighbours))
    else:
        from zadu import MEASURE, make_spec
        from .metrics.preservation import _run_zadu_measure, _ZADU_RNG_LOCK
        for values in (x,mu):
            if len(np.unique(values.astype(np.float32),axis=0)) != len(values):
                raise ValueError('Duplicate float32 coordinates: ZADU cohesiveness unavailable')
        spec = make_spec(MEASURE.SNC,k=config.k,iteration=150,walk_num_ratio=.3,alpha=.1,clustering_strategy='dbscan')
        with _ZADU_RNG_LOCK:
            previous = np.random.get_state()
            try:
                np.random.seed(config.seed)
                result,_ = _run_zadu_measure(spec,x,mu)
            finally:
                np.random.set_state(previous)
    score = float(result[config.metric])
    lower = -config.k/(len(x)-1) if config.metric == 'lcmc' else 0.
    if not np.isfinite(score) or not lower-1e-12 <= score <= 1+1e-12:
        raise ValueError('ZADU did not return a finite preservation score in its expected range')
    return score


def preservation_controller_tag(settings):
    config = PreservationAnnealingConfig(**settings)
    digest = hashlib.sha256(json.dumps(asdict(config),sort_keys=True).encode()).hexdigest()[:10]
    return config.metric+'-'+digest


def make_preservation_loss(settings, config, run_folder, n_valid):
    """Use native training, native loss, native scheduler interface; no new trainer.

    Capture a fixed cohort from the ordered validation loader. Measure at the
    end of warm-up and each stage; changes affect only subsequent epochs.
    Labels and test arrays are not accessed. Final validation is flushed before
    AUTOENCODIX reinitializes the trainer for prediction.
    """
    import torch
    from autoencodix.utils._losses import VarixLoss
    policy = PreservationAnnealingConfig(**settings)
    if config.anneal_function=='no-annealing' or config.default_vae_loss!='kl' or config.loss_reduction!='mean':
        raise ValueError('Preservation annealing needs KL, mean reduction and an enabled annealing adapter')
    count = min(n_valid,policy.n_subsample)
    if count<=2*policy.k:
        raise ValueError('Validation subset must exceed 2*k')
    positions = np.sort(np.random.default_rng(policy.seed).choice(n_valid,count,False))
    folder = Path(run_folder)
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'controller_config.json').write_text(json.dumps(dict(settings=asdict(policy),
        validation_positions=positions.tolist(),n_validation=n_valid,signal='validation X versus mu; no labels'),indent=2)+'\n')

    class PreservationScheduler:
        w = 0.

        def get_weight(self, *, epoch_current, total_epoch, func):
            return self.w

    class PreservationVarixLoss(VarixLoss):
        def __init__(self, config, annealing_scheduler=None):
            if annealing_scheduler is not None:
                raise ValueError('Cannot combine schedulers')
            self.schedule = PreservationScheduler()
            super().__init__(config,annealing_scheduler=self.schedule)
            self.stage = PreservationStage()
            self.epoch = None
            self.offset = 0
            self.inputs, self.means, self.rows = [], [], []
            self.reference = None
            self.logged = set()

        def flush_validation(self):
            if self.epoch is None or self.epoch in self.logged:
                return
            if self.offset!=n_valid:
                raise ValueError('Validation loader did not cover the expected ordered cohort')
            if self.reference is None:
                self.reference = np.concatenate(self.inputs)
                reference = dict(x=self.reference, validation_positions=positions)
                manifest = folder.parent.parent/'split.json'
                if manifest.exists():
                    ids = np.asarray(json.loads(manifest.read_text())['valid'], dtype=str)
                    if len(ids) != n_valid:
                        raise ValueError('Saved validation IDs do not match the ordered loader')
                    reference['sample_ids'] = ids[positions]
                np.savez_compressed(folder/'controller_validation_reference.npz', **reference)
            measure = (self.epoch+1==policy.warmup_epochs or
                       (self.epoch+1>policy.warmup_epochs and
                        (self.epoch+1-policy.warmup_epochs)%policy.step_epochs==0) or
                       self.epoch==config.epochs-1)
            score,warning = np.nan,None
            if measure:
                try:
                    score = measure_preservation(self.reference,np.concatenate(self.means),policy)
                except (ValueError,RuntimeError,KeyError,ZeroDivisionError) as error:
                    warning = f'{type(error).__name__}: {error}'
            if measure and np.isfinite(score) and self.stage.baseline is None and self.epoch+1>=policy.warmup_epochs:
                self.stage = replace(self.stage, baseline=score)
            row = dict(epoch=self.epoch+1,w=self.schedule.w,effective_beta=config.beta*self.schedule.w,
                       beta_over_d=config.beta*self.schedule.w/config.latent_dim,
                       state=self.stage.state,reason=self.stage.reason,metric=policy.metric,metric_value=score,target=preservation_target(self.stage,policy),
                       configured_target=policy.target,target_mode=policy.target_mode,rule=policy.rule,
                       tolerance=policy.tolerance,baseline=self.stage.baseline,
                       status='unavailable' if warning else 'ok' if measure else 'not_measured',warning=warning,
                       seed=config.global_seed,latent_dim=config.latent_dim)
            self.rows.append(row)
            pd.DataFrame(self.rows).to_csv(folder/'controller_log.csv',index=False)
            pd.DataFrame([dict(epoch=r['epoch'],beta_config=r['effective_beta'],beta_over_d=r['beta_over_d'])
                          for r in self.rows]).to_csv(folder/'beta_schedule.csv',index=False)
            self.logged.add(self.epoch)
            # No decision is needed after the last epoch actually trained.
            if measure and self.epoch<config.epochs-1 and self.epoch+1>=policy.warmup_epochs:
                self.stage = advance_preservation_stage(self.stage,score,policy)
                from tqdm.auto import tqdm
                tqdm.write(f'Epoch {self.epoch+1}: {policy.metric}={score:.4f}, target={policy.target:g}; '
                           f'next beta={config.beta*self.stage.w:g}, {self.stage.state} {self.stage.reason}')
            self.offset = 0
            self.inputs, self.means = [], []

        def forward(self, model_output, targets, epoch=None, **kwargs):
            if epoch is not None and torch.is_grad_enabled() and epoch!=self.epoch:
                self.flush_validation()
                self.schedule.w = self.stage.w
                self.epoch = int(epoch)
            if epoch is not None and not torch.is_grad_enabled():
                local = positions[(positions>=self.offset)&(positions<self.offset+len(targets))]-self.offset
                if len(local):
                    if self.reference is None:
                        self.inputs.append(targets[local].detach().cpu().numpy())
                    self.means.append(model_output.latent_mean[local].detach().cpu().numpy())
                self.offset += len(targets)
            result = super().forward(model_output=model_output,targets=targets,epoch=epoch,**kwargs)
            if epoch==config.epochs-1 and not torch.is_grad_enabled() and self.offset==n_valid:
                self.flush_validation()
            return result

    return PreservationVarixLoss
