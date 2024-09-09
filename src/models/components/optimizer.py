import torch
from torch.optim import Adam, SGD, Adagrad, RMSprop
from torch.optim.lr_scheduler import (
    StepLR,
    LinearLR,
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


class TotalStepLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
        total_iters: int = None,
    ) -> None:
        self.total_iters = total_iters
        super(TotalStepLR, self).__init__(optimizer, last_epoch)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self) -> List[float]:
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)
        if self.total_iters is None or self.total_iters == 0:
            return [base_lr for base_lr in self.base_lrs]
        else:
            return [
                base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs
            ]

    def _get_closed_form_lr(self) -> List[float]:
        if self.total_iters is None or self.total_iters == 0:
            return [base_lr for base_lr in self.base_lrs]
        else:
            return [
                base_lr * self.gamma ** (self.last_epoch / self.step_size)
                for base_lr in self.base_lrs
            ]


_SCHEDULERS: Dict[str, torch.optim.lr_scheduler.LRScheduler] = {
    "step": TotalStepLR,
    "linear": LinearLR,
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
