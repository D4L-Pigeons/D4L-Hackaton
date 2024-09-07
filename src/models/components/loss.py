from argparse import Namespace
from typing import Any, Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import ConfigStructure, validate_config_structure
from models.components.misc import (
    TensorAggregator,
    get_tensor_aggregator,
    TensorReductor,
    get_tensor_reductor,
)
from utils.common_types import (
    StructuredLoss,
    format_structured_forward_output,
    format_structured_loss,
    Batch,
    StructuredForwardOutput,
    format_structured_forward_output,
)
from typing import List
from einops import reduce as einops_reduce

_LOSS_NAME_MANAGER: Dict[str, str] = {
    "posterior_neg_entropy": "postr_neg_entr",
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


class LossManager:

    _config_structure: ConfigStructure = {
        "reductions": [
            {"reduce_pattern": str, "reductor_name": str, "kwargs": Namespace}
        ]
    }

    def __init__(self, cfg: Namespace) -> None:
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._sum_tensor_aggregator: TensorAggregator = get_tensor_aggregator(
            aggregation_type="sum", kwargs={}
        )
        self._reduction_specs: List[Namespace] = cfg.reductions
        self._tensor_reductors: Callable[[torch.Tensor], torch.Tensor] = []
        self._setup_reduce_loss_tensor()
        # self._tensor_reductor: TensorReductor = get_tensor_reductor(
        #     tensor_reductor_name=cfg.tensor_reductor.name,
        #     kwargs=vars(cfg.tensor_reductor.kwargs),
        # )
        self._loss_unreduced: List[torch.Tensor] = []
        self._loss_reduced: torch.Tensor = torch.zeros(1)
        self._loss_log_dict: Dict[str, float] = {}

    @property
    def log_dict(self) -> Dict[str, float]:
        return self._loss_log_dict

    def _setup_reduce_loss_tensor(self) -> None:
        for reduction_spec in self._reduction_specs:
            self._tensor_reductors.append(
                get_tensor_reductor(
                    tensor_reductor_name=reduction_spec.reductor_name,
                    kwargs=vars(reduction_spec.kwargs),
                )
            )

    def reset(self) -> None:
        self._loss_unreduced: List[torch.Tensor] = []
        self._loss_reduced: torch.Tensor = torch.zeros(1)
        self._loss_log_dict: Dict[str, float] = {}

    def _add_to_log_dict(self, loss_name: str, loss: torch.Tensor) -> None:
        self._loss_log_dict[loss_name] = loss.item()

    def _sum_unreduced(self, loss: torch.Tensor) -> None:
        self._loss_unreduced = [
            self._sum_tensor_aggregator(self._loss_unreduced + [loss])
        ]

    def process_structured_loss(self, structured_loss: StructuredLoss) -> None:
        # print(
        #     f"PROCESSING LOSS: '{structured_loss['name']}' | REDUCED: {structured_loss['reduced']}"
        # )
        if structured_loss["reduced"]:
            self._loss_reduced += structured_loss["coef"] * structured_loss["data"]
            self._add_to_log_dict(
                loss_name=structured_loss["name"], loss=structured_loss["data"]
            )
        else:
            self._sum_unreduced(loss=structured_loss["coef"] * structured_loss["data"])
            self._add_to_log_dict(
                loss_name=structured_loss["name"], loss=structured_loss["data"].mean()
            )

    def _reduce_unreduced(self) -> torch.Tensor:
        loss = self._loss_unreduced[0]
        # print(f"loss_unnagregated.shape = {loss.shape}")

        for reduction_spec, tensor_reductor in zip(
            self._reduction_specs, self._tensor_reductors
        ):
            loss = einops_reduce(
                tensor=loss,
                pattern=reduction_spec.reduce_pattern,
                reduction=tensor_reductor,
            )
        assert (
            loss.shape == ()
        ), f"Loss shape after aggregation should be () got {loss.shape}."
        # print(f"loss_unnagregated.shape = {loss_unnagregated.shape}")

        return loss

    def get_loss(self) -> torch.Tensor:
        loss: torch.Tensor = self._loss_reduced

        if len(self._loss_unreduced) != 0:
            loss += self._reduce_unreduced()

        self.reset()
        return loss


class ReconstructionLoss(nn.Module):
    _config_structure: ConfigStructure = {
        "coef": float,
        "loss_fn": {"name": str, "kwargs": Namespace},
        "tensor_reductor": {"name": str, "kwargs": Namespace},
    }

    def __init__(self, cfg: Namespace) -> None:
        super(ReconstructionLoss, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._coef: float = cfg.coef
        self._loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
            get_reconstruction_loss(
                loss_name=cfg.loss_fn.name, kwargs=vars(cfg.loss_fn.kwargs)
            )
        )

        self._tensor_reductor: TensorReductor = get_tensor_reductor(
            tensor_reductor_name=cfg.tensor_reductor.name,
            kwargs=vars(cfg.tensor_reductor.kwargs),
        )

    def forward(
        self, batch: Batch, data_name: str, target_name: str, reduce_pattern: str
    ) -> StructuredForwardOutput:

        loss: torch.Tensor = self._loss_fn(batch[data_name], batch[target_name])

        loss = einops_reduce(
            tensor=loss, pattern=reduce_pattern, reduction=self._tensor_reductor
        )

        reduced = loss.shape == (1,)  # check if shape is torch.Size([1])
        # print(f"RECONSTRUCTION LOSS SHAPE: {loss.shape} | REDUCED = {reduced}")

        return format_structured_forward_output(
            batch=batch,
            losses=[
                format_structured_loss(
                    loss=loss,
                    coef=self._coef,
                    name=map_loss_name(loss_name="reconstruction"),
                    reduced=reduced,
                )
            ],
        )
