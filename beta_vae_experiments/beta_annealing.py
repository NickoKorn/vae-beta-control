"""Linear beta warm-up using the native AUTOENCODIX loss extension point."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class LinearBetaWarmup:
    """Start at beta=0, reach target at the last warm-up epoch, then hold.

    Epochs are zero-based internally. With warmup_epochs=300, factors are
    0 at index 0 and 1 at index 299; indices 300 onward retain factor 1.
    This fixed schedule is not a feedback controller or a collapse guarantee.
    """

    epochs: int = 1000
    warmup_epochs: int = 300
    kind: str = 'linear_warmup'

    def __post_init__(self):
        if self.kind != 'linear_warmup':
            raise ValueError('Only linear_warmup is supported')
        if any(type(value) is not int for value in (self.epochs, self.warmup_epochs)):
            raise ValueError('epochs and warmup_epochs must be integers')
        if not 2 <= self.warmup_epochs <= self.epochs:
            raise ValueError('Require 2 <= warmup_epochs <= epochs')

    def factors(self):
        return np.minimum(np.arange(self.epochs) / (self.warmup_epochs - 1), 1.)


def schedule_from_settings(settings, beta_high_over_d):
    if not np.isfinite(beta_high_over_d) or beta_high_over_d <= 0:
        raise ValueError('Target beta_over_d must be finite and positive')
    unknown = set(settings) - {'kind', 'epochs', 'warmup_epochs'}
    if unknown:
        raise ValueError(f'Unknown beta_schedule keys: {sorted(unknown)}')
    return LinearBetaWarmup(**settings)


def schedule_tag(settings):
    encoded = json.dumps(settings, sort_keys=True, separators=(',', ':')).encode()
    return settings['kind'] + '-' + hashlib.sha256(encoded).hexdigest()[:10]


def make_scheduled_loss(settings, config):
    """Inject a scheduler through the supported native loss extension point."""
    from autoencodix.utils._losses import VarixLoss
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

    class ScheduledVarixLoss(VarixLoss):
        def __init__(self, config, annealing_scheduler=None):
            if annealing_scheduler is not None:
                raise ValueError('Cannot combine two annealing schedulers')
            super().__init__(config, annealing_scheduler=FixedScheduler())

    return ScheduledVarixLoss
