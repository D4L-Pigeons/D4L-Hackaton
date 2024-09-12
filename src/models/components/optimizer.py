r"""
This module provides functionality for creating and managing optimizers and learning rate schedulers for PyTorch models.

Classes:
    TotalStepLR: A learning rate scheduler that decays the learning rate of each parameter group by a specified gamma every step_size epochs. It also supports a total_iters parameter to control the total number of iterations.

Functions:
    get_optimizer(optimizer_name: str, params: Iterable[torch.nn.Parameter], kwargs: Dict[str, Any]) -> torch.optim.Optimizer:

    get_scheduler(scheduler_name: str, optimizer: torch.optim.Optimizer, kwargs: Dict[str, Any]) -> torch.optim.lr_scheduler.LRScheduler:

    get_chained_scheduler(optimizer: torch.optim.Optimizer, schedulers_cfg: List[Namespace]) -> ChainedScheduler:

Constants:
    _OPTIMIZERS: A dictionary mapping optimizer names to their corresponding PyTorch optimizer classes.
    _SCHEDULERS: A dictionary mapping scheduler names to their corresponding PyTorch learning rate scheduler classes.

"""

import torch
from torch.optim import Adam, SGD, Adagrad, RMSprop
from torch.optim.lr_scheduler import (
    LinearLR,
    ConstantLR,
    ChainedScheduler,
)
from argparse import Namespace
from typing import Dict, Any, Iterable, List

_OPTIMIZERS: Dict[str, torch.optim.Optimizer] = {
    "adam": Adam,
    "sgd": SGD,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
}


class TotalStepLR(torch.optim.lr_scheduler.LRScheduler):
    r"""
    TotalStepLR is a learning rate scheduler that decays the learning rate of each parameter group by gamma every step_size epochs.
    It also supports a total_iters parameter to control the total number of iterations.

    Args:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float, optional): Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch (int, optional): The index of the last epoch. Default: -1.
        total_iters (int, optional): Total number of iterations. Default: None.

    Methods:
        get_lr() -> List[float]:
            Compute the learning rate for the current epoch.

        _get_closed_form_lr() -> List[float]:
            Compute the learning rate using the closed form of the learning rate schedule.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
        total_iters: int = None,
    ) -> None:
        super(TotalStepLR, self).__init__(optimizer, last_epoch)

        self.total_iters = total_iters
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self) -> List[float]:
        r"""
        Get the current learning rates for each parameter group.

        This method calculates the learning rate for each parameter group based on
        the base learning rate, the gamma value, the current epoch, and the step size.
        If the total number of iterations is not set or is zero, it returns the base
        learning rates.

        Returns:
            List[float]: A list of learning rates for each parameter group.
        """

        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)
        if self.total_iters is None or self.total_iters == 0:
            return [base_lr for base_lr in self.base_lrs]
        else:
            return [
                base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs
            ]

    def _get_closed_form_lr(self) -> List[float]:
        r"""
        Calculate the learning rates using a closed-form solution.

        This method computes the learning rates based on the current epoch and
        the specified step size and gamma. If the total number of iterations is
        not defined or is zero, it returns the base learning rates.

        Returns:
            List[float]: A list of learning rates for each parameter group.
        """

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
    r"""
    Retrieves an optimizer from the available optimizers based on the provided name.

    Args:
        optimizer_name (str): The name of the optimizer to retrieve.
        params (Iterable[torch.nn.Parameter]): The parameters to be optimized.
        kwargs (Dict[str, Any]): Additional keyword arguments to pass to the optimizer.

    Returns:
        torch.optim.Optimizer: An instance of the requested optimizer.

    Raises:
        ValueError: If the provided optimizer_name is not found in the available optimizers.
    """

    optimizer_class = _OPTIMIZERS.get(optimizer_name, None)

    if optimizer_class is None:
        raise ValueError(
            f"The provided optimizer_name {optimizer_name} is invalid. Must be one of {list(_OPTIMIZERS.keys())}"
        )

    return optimizer_class(params=params, **kwargs)


def get_scheduler(
    scheduler_name: str, optimizer: torch.optim.Optimizer, kwargs: Dict[str, Any]
) -> torch.optim.lr_scheduler.LRScheduler:
    r"""
    Retrieves a learning rate scheduler based on the provided scheduler name.

    Args:
        scheduler_name (str): The name of the scheduler to retrieve.
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        kwargs (Dict[str, Any]): Additional keyword arguments to pass to the scheduler.

    Returns:
        torch.optim.lr_scheduler.LRScheduler: The learning rate scheduler instance.

    Raises:
        ValueError: If the provided scheduler_name is not found in the available schedulers.
    """

    scheduler_class = _SCHEDULERS.get(scheduler_name, None)

    if scheduler_class is None:
        raise ValueError(
            f"The provided scheduler_name {scheduler_name} is invalid. Must be one of {list(_SCHEDULERS.keys())}"
        )

    return scheduler_class(optimizer=optimizer, **kwargs)


def get_chained_scheduler(
    optimizer: torch.optim.Optimizer, schedulers_cfg: List[Namespace]
) -> ChainedScheduler:
    r"""
    Creates a ChainedScheduler from a list of scheduler configurations.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which the schedulers will be chained.
        schedulers_cfg (List[Namespace]): A list of Namespace objects, each containing the name and
                                          kwargs for a scheduler.

    Returns:
        ChainedScheduler: A chained scheduler composed of the individual schedulers specified in
                          schedulers_cfg.
    """

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
