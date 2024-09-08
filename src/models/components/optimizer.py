import torch
from torch.optim import Adam, SGD, Adagrad, RMSprop
from torch.optim.lr_scheduler import (
    StepLR,
    LinearLR,
    ExponentialLR,
    ConstantLR,
    ChainedScheduler,
)
from typing import Dict, Any, Iterable, List
from argparse import Namespace

_OPTIMIZERS: Dict[str, torch.optim.Optimizer] = {
    "adam": Adam,
    "sgd": SGD,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
}

_SCHEDULERS: Dict[str, torch.optim.lr_scheduler.LRScheduler] = {
    "step": StepLR,
    "linear": LinearLR,
    "exp": ExponentialLR,
    "const": ConstantLR,
}


def get_optimizer(
    optimizer_name: str, params: Iterable[torch.nn.Parameter], kwargs: Dict[str, Any]
) -> torch.optim.Optimizer:
    optimizer_class = _OPTIMIZERS.get(optimizer_name, None)
    if optimizer_class is not None:
        return optimizer_class(params=params, **kwargs)
    raise ValueError(
        f"The provided optimizer_name {optimizer_name} is invalid. Must be one of {list(_OPTIMIZERS.keys())}"
    )


def get_scheduler(
    scheduler_name: str, optimizer: torch.optim.Optimizer, kwargs: Dict[str, Any]
) -> torch.optim.lr_scheduler.LRScheduler:
    scheduler_class = _SCHEDULERS.get(scheduler_name, None)
    if scheduler_class is not None:
        return scheduler_class(optimizer=optimizer, **kwargs)
    raise ValueError(
        f"The provided scheduler_name {scheduler_name} is invalid. Must be one of {list(_SCHEDULERS.keys())}"
    )


def get_chained_scheduler(
    optimizer: torch.optim.Optimizer, schedulers_cfg: List[Namespace]
) -> ChainedScheduler:
    schedulers = [
        get_scheduler(
            scheduler_name=scheduler_cfg.name,
            optimizer=optimizer,
            kwargs=vars(scheduler_cfg.kwargs),
        )
        for scheduler_cfg in schedulers_cfg
    ]

    chained_scheduler = ChainedScheduler(schedulers=schedulers)

    return chained_scheduler
