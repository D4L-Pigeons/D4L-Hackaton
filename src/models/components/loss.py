from argparse import Namespace
from typing import Callable, Dict, TypeAlias

import torch
import torch.nn.functional as F

StructuredLoss: TypeAlias = Dict[str, torch.Tensor | str | bool]

_LOSS_NAME_MANAGER: Dict[str, str] = {
    "posterior_entropy": "postr_H",
    "prior_nll": "prior_nll",
    "latent_constraint": "lat_constr",
    "latent_fuzzy_clustering": "fuzz_clust",
    "clustering_component_reg": "comp_clust_reg",
}


def map_loss_name(loss_name: str) -> str:
    return _LOSS_NAME_MANAGER[loss_name]


_INVERSE_NAME_MANAGER: Dict[str, str] = {
    val: key for key, val in _LOSS_NAME_MANAGER.items()
}


def map_loss_name_inverse(loss_name: str) -> str:
    return _INVERSE_NAME_MANAGER[loss_name]


_EXPLICIT_CONSTRAINT: Dict[str, Callable[[torch.Tensor, int | None], torch.Tensor]] = {
    "l2": lambda x, dim: x.pow(2).mean(dim=dim),
    "huber": lambda x, dim: F.huber_loss(
        x, torch.zeros_like(x), delta=1.0, reduction="none"
    ).mean(dim=dim),
}


def get_explicit_constraint(
    constraint_name: str,
) -> Callable[[torch.Tensor, int | None], torch.Tensor]:
    return _EXPLICIT_CONSTRAINT[constraint_name]


def format_loss(loss: torch.Tensor, name: str, aggregated: bool) -> StructuredLoss:
    return {
        "data": loss,
        "name": name,
        "aggregated": aggregated,
    }


_LOSS_AGGREGATORS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "mean": lambda x: x.mean(),
    "logsumexp": lambda x: x.logsumexp(dim=0),
}


def get_loss_aggregator(cfg: Namespace) -> Callable[[torch.Tensor], torch.Tensor]:
    return _LOSS_AGGREGATORS[cfg.name]
