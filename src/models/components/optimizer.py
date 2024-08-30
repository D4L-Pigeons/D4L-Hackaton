import torch
from torch.optim import Optimizer, Adam, SGD, Adagrad, RMSprop
from typing import Dict, Any, Iterable

_OPTIMIZERS: Dict[str, Optimizer] = {
    "adam": Adam,
    "sgd": SGD,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
}


def get_optimizer(
    optimizer_name: str, params: Iterable[torch.nn.Parameter], kwargs: Dict[str, Any]
) -> Optimizer:
    optimizer_class = _OPTIMIZERS.get(optimizer_name, None)
    if optimizer_class is not None:
        return optimizer_class(params=params, **kwargs)
    raise ValueError(
        f"The provided optimizer_name {optimizer_name} is invalid. Must be one of {_OPTIMIZERS.keys()}"
    )
