from argparse import Namespace
from typing import Callable, Dict
import torch
import torch.nn.functional as F


_LOSS_NAME_MANAGER: Dict[str, str] = {
    "posterior_entropy": "postr_H",
    "prior_nll": "prior_nll",
    "latent_constraint": "lat_constr",
    "latent_fuzzy_clustering": "fuzz_clust",
    "clustering_component_reg": "comp_clust_reg",
}


def map_loss_name(loss_name: str) -> str:
    loss = _LOSS_NAME_MANAGER.get(loss_name, None)
    if loss is not None:
        return loss
    raise ValueError(
        f"The provided loss_name {loss_name} is wrong. Must be one of {' ,'.join(list(_LOSS_NAME_MANAGER.keys()))}"
    )


_INVERSE_LOSS_NAME_MANAGER: Dict[str, str] = {
    val: key for key, val in _LOSS_NAME_MANAGER.items()
}


def map_loss_name_inverse(loss_name: str) -> str:
    loss = _INVERSE_LOSS_NAME_MANAGER.get(loss_name, None)
    if loss is not None:
        return loss
    raise ValueError(
        f"The provided loss_name {loss_name} is wrong. Must be one of {' ,'.join(list(_INVERSE_LOSS_NAME_MANAGER.keys()))}"
    )


_EXPLICIT_CONSTRAINT: Dict[str, Callable[[torch.Tensor, int | None], torch.Tensor]] = {
    "l2": lambda x, dim: x.pow(2).mean(dim=dim),
    "huber": lambda x, dim: F.huber_loss(
        x, torch.zeros_like(x), delta=1.0, reduction="none"
    ).mean(dim=dim),
}


def get_explicit_constraint(
    constraint_name: str,
) -> Callable[[torch.Tensor, int | None], torch.Tensor]:
    constraint = _EXPLICIT_CONSTRAINT.get(constraint_name, None)
    if constraint is not None:
        return constraint
    raise ValueError(
        f"The provided constraint_name {constraint_name} is wrong. Must be one of {' ,'.join(list(_EXPLICIT_CONSTRAINT.keys()))}"
    )


_RECONSTRUCTION_LOSSES: Dict[
    str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
] = {
    "mse": F.mse_loss,
    "l1": F.l1_loss,
    "smooth_l1": F.smooth_l1_loss,
}


def get_reconstruction_loss(
    loss_name: str,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    loss = _RECONSTRUCTION_LOSSES.get(loss_name, None)
    if loss is not None:
        return loss
    raise ValueError(
        f"The provided loss_name {loss_name} is wrong. Must be one of {' ,'.join(list(_RECONSTRUCTION_LOSSES.keys()))}"
    )


_LOSS_AGGREGATORS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "mean": lambda x: x.mean(),
    "logsumexp": lambda x: x.logsumexp(dim=0),
}


def get_loss_aggregator(
    loss_aggregator_name: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    loss_aggregator = _LOSS_AGGREGATORS.get(loss_aggregator, None)
    if loss_aggregator is not None:
        return loss_aggregator
    raise ValueError(
        f"The provided loss_aggregator_name {loss_aggregator_name} is wrong. Must be one of {' ,'.join(list(_LOSS_AGGREGATORS.keys()))}"
    )
