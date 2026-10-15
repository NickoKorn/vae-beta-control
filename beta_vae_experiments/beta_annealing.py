"""Fixed, JSON-controlled beta cycles with long high/low plateaus.

Uses the native AUTOENCODIX loss; only its annealing factor is supplied here.
All settings refer to epochs. beta_low_over_d uses the existing mean-KL units.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class EpochBetaSchedule:
    kind: str
    low_fraction: float
    initial_hold: int
    cycles: int
    ramp_up: int
    high_hold: int
    ramp_down: int
    low_hold: int

    def __post_init__(self):
        if self.kind not in {'constant_low', 'constant_high', 'step', 'linear', 'cosine'}:
            raise ValueError(f'Unknown schedule {self.kind}')
        if not np.isfinite(self.low_fraction) or not 0 <= self.low_fraction <= 1:
            raise ValueError('Require 0 <= beta_low / beta_high <= 1')
        for key in ['initial_hold', 'cycles', 'ramp_up', 'high_hold', 'ramp_down', 'low_hold']:
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{key} must be a nonnegative integer')
        if self.cycles < 1 or self.ramp_up < 2 or self.ramp_down < 2 or self.high_hold < 1 or self.low_hold < 1:
            raise ValueError('Require cycles >=1, ramps >=2, and nonempty plateaus')

    @property
    def epochs(self):
        return self.initial_hold + self.cycles * (self.ramp_up + self.high_hold + self.ramp_down + self.low_hold)

    def factors(self):
        if self.kind.startswith('constant_'):
            value = self.low_fraction if self.kind == 'constant_low' else 1.
            return np.full(self.epochs, value)
        up = np.linspace(0., 1., self.ramp_up)
        down = np.linspace(1., 0., self.ramp_down)
        if self.kind == 'cosine':
            up, down = (1 - np.cos(np.pi * up)) / 2, (1 - np.cos(np.pi * down)) / 2
        elif self.kind == 'step':
            up, down = np.ones(self.ramp_up), np.zeros(self.ramp_down)
        cycle = np.concatenate([up, np.ones(self.high_hold), down, np.zeros(self.low_hold)])
        fractions = np.concatenate([np.zeros(self.initial_hold), np.tile(cycle, self.cycles)])
        return self.low_fraction + (1 - self.low_fraction) * fractions


def schedule_from_settings(settings, beta_high_over_d):
    settings = dict(settings)
    if set(settings) - {'kind', 'beta_low_over_d', 'initial_hold', 'cycles', 'ramp_up', 'high_hold', 'ramp_down', 'low_hold'}:
        raise ValueError('Unknown beta_schedule keys')
    low = float(settings.pop('beta_low_over_d'))
    high = float(beta_high_over_d)
    if not np.isfinite(high) or high <= 0 or not np.isfinite(low) or low < 0 or low > high:
        raise ValueError('Require 0 <= beta_low_over_d <= beta_high_over_d and high > 0')
    return EpochBetaSchedule(low_fraction=low / high, **settings)


def schedule_tag(settings):
    encoded = json.dumps(settings, sort_keys=True, separators=(',', ':')).encode()
    return settings['kind'] + '-' + hashlib.sha256(encoded).hexdigest()[:10]


def make_scheduled_loss(settings, config):
    """Inject a scheduler through the supported native loss extension point."""
    from beta_vae_experiments.run_artifacts import RecordedVarixLoss
    if config.default_vae_loss != 'kl' or config.loss_reduction != 'mean':
        raise ValueError('Fixed beta_over_d schedules require KL and mean reduction')
    if config.anneal_function == 'no-annealing':
        raise ValueError('Native no-annealing bypasses custom schedulers; use 3phase-linear as the enabled adapter setting')
    schedule = schedule_from_settings(settings, config.beta / config.latent_dim)
    if config.epochs != schedule.epochs:
        raise ValueError(f'epochs={config.epochs} must equal schedule duration {schedule.epochs}')
    factors = schedule.factors()

    class FixedScheduler:
        def get_weight(self, *, epoch_current, total_epoch, func):
            if total_epoch != len(factors):
                raise ValueError('Epoch count differs from the configured fixed schedule')
            index = len(factors) - 1 if epoch_current is None else int(epoch_current)
            if index < 0 or index >= len(factors):
                raise ValueError(f'Epoch outside schedule: {index}')
            return float(factors[index])

    class ScheduledVarixLoss(RecordedVarixLoss):
        def __init__(self, config, annealing_scheduler=None):
            if annealing_scheduler is not None:
                raise ValueError('Cannot combine two annealing schedulers')
            super().__init__(config, annealing_scheduler=FixedScheduler())

    return ScheduledVarixLoss
