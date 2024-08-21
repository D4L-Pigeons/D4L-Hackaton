from argparse import Namespace
from typing import Any, Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.config import ConfigStructure, validate_config_structure
from src.models.components.misc import (
    AggregateDataAdapter,
    get_tensor_aggregator,
    TensorAggregator,
)
from src.utils.common_types import StructuredLoss
from typing import TypeAlias, List

_LOSS_NAME_MANAGER: Dict[str, str] = {
    "posterior_entropy": "postr_H",
    "prior_nll": "prior_nll",
    "latent_constraint": "lat_constr",
    "latent_fuzzy_clustering": "fuzz_clust",
    "clustering_component_reg": "comp_clust_reg",
    "reconstruction": "reconstr",
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
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "smooth_l1": nn.SmoothL1Loss,
}


def get_reconstruction_loss(
    loss_name: str,
    kwargs: Dict[str, Any],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    loss = _RECONSTRUCTION_LOSSES.get(loss_name, None)
    if loss is not None:
        return loss(**kwargs)
    raise ValueError(
        f"The provided loss_name {loss_name} is wrong. Must be one of {' ,'.join(list(_RECONSTRUCTION_LOSSES.keys()))}"
    )


LossAggregator: TypeAlias = Callable[[torch.Tensor], torch.Tensor]

_LOSS_AGGREGATORS: Dict[str, LossAggregator] = {
    "mean": lambda x: x.mean(),
    "logsumexpmean": lambda x: x.logsumexp(
        dim=1
    ).mean(),  # dim=1 corresponds to latent samples
}


def get_loss_aggregator(
    loss_aggregator_name: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    loss_aggregator = _LOSS_AGGREGATORS.get(loss_aggregator_name, None)
    if loss_aggregator is not None:
        return loss_aggregator
    raise ValueError(
        f"The provided loss_aggregator_name {loss_aggregator_name} is wrong. Must be one of {' ,'.join(list(_LOSS_AGGREGATORS.keys()))}"
    )


class LossManager:
    r"""
    Class representing a loss manager used for aggregating the losses and outputting the loss terms to the logger.

    Args:
        cfg (Namespace): The configuration namespace.

    Attributes:
        _config_structure (ConfigStructure): The configuration structure.
        _sum_tensor_aggregator (TensorAggregator): The tensor aggregator for summing tensors.
        _loss_aggregator (LossAggregator): The loss aggregator.
        _loss_unaggregated (List[torch.Tensor]): The list of unaggregated losses.
        _loss_aggregate (None | torch.Tensor): The aggregated loss.
        _loss_log_dict (Dict[str, float]): The dictionary for logging losses.

    Methods:
        reset(): Resets the loss manager.
        _add_to_log_dict(loss_name: str, loss: torch.Tensor): Adds a loss to the log dictionary.
        _sum_unaggregated(loss: torch.Tensor): Sums an unaggregated loss.
        process_structured_loss(structured_loss: StructuredLoss): Processes a structured loss.
        get_loss_aggregate(): Returns the aggregated loss.

    """

    _config_structure: ConfigStructure = {"loss_aggregator": str}

    def __init__(self, cfg: Namespace) -> None:
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._sum_tensor_aggregator: TensorAggregator = get_tensor_aggregator(
            aggregation_type="sum", kwargs={}
        )
        self._loss_aggregator: LossAggregator = get_loss_aggregator(
            loss_aggregator_name=cfg.loss_aggregator
        )
        self._loss_unaggregated: List[torch.Tensor] = []
        self._loss_aggregate: torch.Tensor = torch.zeros(1)
        self._loss_log_dict: Dict[str, float] = {}

    @property
    def log_dict(self) -> Dict[str, float]:
        return self._loss_log_dict

    def reset(self) -> None:
        self._loss_unaggregated: List[torch.Tensor] = []
        self._loss_aggregate: torch.Tensor = torch.zeros(1)
        self._loss_log_dict: Dict[str, float] = {}

    def _add_to_log_dict(self, loss_name: str, loss: torch.Tensor) -> None:
        self._loss_log_dict[loss_name] = loss.item()

    def _sum_unaggregated(self, loss: torch.Tensor) -> None:
        self._loss_unaggregated = [
            self._sum_tensor_aggregator(self._loss_unaggregated + [loss])
        ]

    def process_structured_loss(self, structured_loss: StructuredLoss) -> None:
        if structured_loss["aggregated"]:
            self._loss_aggregate += structured_loss["coef"] * structured_loss["data"]
            self._add_to_log_dict(
                loss_name=structured_loss["name"], loss=structured_loss["data"]
            )
        else:
            self._sum_unaggregated(
                loss=structured_loss["coef"] * structured_loss["data"]
            )
            self._add_to_log_dict(
                loss_name=structured_loss["name"], loss=structured_loss["data"].mean()
            )

    def get_loss_aggregate(self) -> torch.Tensor:
        loss_aggregate: torch.Tensor = self._loss_aggregate
        if len(self._loss_unaggregated) != 0:
            loss_unnagregated = self._loss_unaggregated[0]
            loss_aggregate += self._loss_aggregator(loss_unnagregated)

        self.reset()
        return loss_aggregate
