# Copyright (c) CAIRI AI Lab. All rights reserved

from .ema_hook import EMAHook, SwitchEMAHook
from .hooks import Hook, Priority, get_priority
from .metrics import metric
from .recorder import Recorder
from .optim_scheduler import get_optim_scheduler
from .optim_constant import optim_parameters
from .save_best_hook import SaveBestHook
from .wandb_hook import WandBHook

hook_maps = {
    'ema_hook': EMAHook,
    'save_best_hook': SaveBestHook,
    'wandb_hook': WandBHook,
    **dict.fromkeys(['semahook', 'switchemahook'], SwitchEMAHook),
}

__all__ = [
    'Hook', 'EMAHook', 'SwitchEMAHook', 'Priority', 'get_priority', 'metric',
    'Recorder', 'get_optim_scheduler', 'optim_parameters'
]