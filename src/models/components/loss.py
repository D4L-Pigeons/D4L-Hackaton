from argparse import Namespace
from typing import Any, Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List
from einops import reduce as einops_reduce

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


class BaseLossCoefScheduler:
    def __init__(
        self, start_coef_scale: float, end_coef_scale: float, num_epochs: int
    ) -> None:
        self._start_coef_scale: float = start_coef_scale
        # self._current_coef_scale: float = start_coef_scale
        self._end_coef_scale: float = end_coef_scale
        self.epoch_count: int = 0
        self.num_epochs: int = num_epochs

    def step(self) -> float:
        """Increase the epoch and return the updated coefficient."""
        self.epoch_count += 1
        self._current_coef_scale = self.get_coef_scale()
        return self.get_coef_scale()

    def get_coef_scale(self) -> float:
        """This method should be implemented in subclasses."""
        raise NotImplementedError

    # THIS WOULD BE COOL TO IMPLEMENT ALONG WITH USE OF self._current_coef_scale NOT TO CALCULATE EVERYTHIN FROM SCRATCH EACH TIME AS DONE IN LR SCHEDULERS IN PYTORCH BUT I GUESS IT HAS MARGINAL SIGNIFICANCE
    # get_coef_scale is called once per batch for each loss coef scheduler calculations could be reused
    # def _get_closed_form_coef(self) -> float:
    #     """This method should be implemented in subclasses."""
    #     raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        """Save the current state."""
        return {"epoch_count": self.epoch_count}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the saved state."""
        self.epoch_count = state_dict["epoch_count"]
        # self._current_coef_scale = self._get_closed_form_coef()


class LinearLossCoefScheduler(BaseLossCoefScheduler):

    def get_coef_scale(self) -> float:
        return self._start_coef_scale + (
            self._end_coef_scale - self._start_coef_scale
        ) * (self.epoch_count / self.num_epochs)


class ExponentialLossCoefScheduler(BaseLossCoefScheduler):

    def get_coef_scale(self) -> float:
        decay_rate = (self._end_coef_scale / self._start_coef_scale) ** (
            1 / self.num_epochs
        )
        return self._start_coef_scale * (decay_rate**self.epoch_count)


# NOTE: The below ChainedLossCoefScheduler scheduler has different | better :) logic than ChainedScheduler from pytorch.
class ChainedLossCoefScheduler:
    def __init__(self, schedulers: List[Any]):
        """
        Args:
            schedulers (list): A list of scheduler instances that will be applied sequentially.
        """
        self.schedulers: List[BaseLossCoefScheduler] = schedulers
        self.current_scheduler_idx: int = 0

    def step(self) -> float:
        """Step through the current scheduler. If it's done, move to the next scheduler."""
        current_scheduler = self.schedulers[self.current_scheduler_idx]
        coef_scale: float = current_scheduler.step()

        # Check if the current scheduler is done
        if current_scheduler.epoch_count >= current_scheduler.num_epochs:
            if self.current_scheduler_idx < len(self.schedulers) - 1:
                self.current_scheduler_idx += 1

        return coef_scale

    def get_coef_scale(self) -> float:
        """
        Returns:
            float: The coefficient scale value.
        """
        current_scheduler = self.schedulers[self.current_scheduler_idx]
        coef_scale = current_scheduler.get_coef_scale()
        return coef_scale

    def state_dict(self) -> Dict[str, Any]:
        """Save the state of all schedulers."""
        return {
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
            "current_scheduler_idx": self.current_scheduler_idx,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the state of all schedulers."""
        for scheduler, state in zip(self.schedulers, state_dict["schedulers"]):
            scheduler.load_state_dict(state)
        self.current_scheduler_idx = state_dict["current_scheduler_idx"]


_LOSS_COEF_SCHEDULERS: Dict[str, BaseLossCoefScheduler] = {
    "linear": LinearLossCoefScheduler,
    "exp": ExponentialLossCoefScheduler,
}


def _get_loss_coef_scheduler(
    scheduler_name: str, kwargs: Dict[str, Any]
) -> BaseLossCoefScheduler:
    coef_scheduler_class = _LOSS_COEF_SCHEDULERS.get(scheduler_name, None)
    if coef_scheduler_class is not None:
        return coef_scheduler_class(**kwargs)
    raise ValueError(
        f"The provided scheduler_name {coef_scheduler_class} is invalid. Must be one of {list(_LOSS_COEF_SCHEDULERS.keys())}"
    )


def _get_chained_loss_coef_scheduler(
    schedulers_cfg: List[Namespace],
) -> ChainedLossCoefScheduler:

    schedulers = [
        _get_loss_coef_scheduler(
            scheduler_name=scheduler_cfg.name,
            kwargs=vars(scheduler_cfg.kwargs),
        )
        for scheduler_cfg in schedulers_cfg
    ]

    chained_scheduler = ChainedLossCoefScheduler(schedulers=schedulers)

    return chained_scheduler


class LossManager:

    _config_structure: ConfigStructure = {
        "reductions": [
            {
                "reduce_pattern": str,
                "reductor_name": str,
                "pre_scale": None | float,
                "post_scale": None | float,
                "kwargs": Namespace,
            }
        ],
        "loss_coef_schedulers": (
            None,
            [
                {
                    "link_name": str,
                    "loss_name": str,
                    "schedulers_cfg": {"name": str, "kwargs": Namespace},
                }
            ],
        ),
    }

    def __init__(self, cfg: Namespace) -> None:
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        # Setting a function used for summing up the loss terms before reducing them to a scalar
        self._sum_tensor_aggregator: TensorAggregator = get_tensor_aggregator(
            aggregation_type="sum", kwargs={}
        )

        # Setting specification of reduction to a scalar of the summed loss terms
        self._reduction_specs: List[Namespace] = cfg.reductions
        self._tensor_reductors: Callable[[torch.Tensor], torch.Tensor] = []
        self._setup_reduce_loss_tensor()

        # Initializing loss term placeholders
        self._loss_unreduced: List[torch.Tensor] = []
        self._loss_reduced: torch.Tensor = torch.zeros(1, dtype=torch.float32)

        # Initialize logging dict used in lightning log_dict
        self._loss_log_dict: Dict[str, float] = {}

        # Setting loss coefficient schedulers if provided
        self._loss_coef_log_dict: Dict[str, float] = {}
        self._loss_coef_schedulers: Dict[str, ChainedLossCoefScheduler] = (
            {
                self._format_full_loss_name(
                    link_name=scheduler_spec.link_name,
                    loss_name=scheduler_spec.loss_name,
                ): _get_chained_loss_coef_scheduler(
                    schedulers_cfg=scheduler_spec.schedulers_cfg
                )
                for scheduler_spec in cfg.loss_coef_schedulers
            }
            if cfg.loss_coef_schedulers is not None
            else {}
        )

    @property
    def loss_log_dict(self) -> Dict[str, float]:
        return self._loss_log_dict

    @property
    def loss_coef_log_dict(self) -> Dict[str, float]:
        return self._loss_coef_log_dict

    def _setup_reduce_loss_tensor(self) -> None:
        for reduction_spec in self._reduction_specs:
            self._tensor_reductors.append(
                get_tensor_reductor(
                    tensor_reductor_name=reduction_spec.reductor_name,
                    kwargs=vars(reduction_spec.kwargs),
                )
            )

    def reset(self) -> None:
        self._loss_unreduced = []
        self._loss_reduced = torch.zeros(1, dtype=torch.float32)
        self._loss_log_dict = {}
        self._loss_coef_log_dict = {}

    def _add_loss_to_loss_log_dict(
        self, full_loss_name: str, loss: torch.Tensor
    ) -> None:
        assert (
            "." not in full_loss_name
        ), f"Character '.' cannot occur in the link_name as it is used in logic for disemabiguating link loss calculation number."

        loss_log_name = full_loss_name

        if loss_log_name in self._loss_log_dict:
            loss_log_name += ".1"
            while loss_log_name in self._loss_log_dict:
                splitted = loss_log_name.split(".")
                loss_log_name = splitted(-2) + "." + str(int(splitted(-1)) + 1)

        self._loss_log_dict[loss_log_name] = loss.item()

    def _add_scaled_loss_coef_to_loss_coef_log_dict(
        self, full_loss_name: str, scaled_coef: float
    ) -> None:
        coef_name = full_loss_name + "_loss_coef"
        self._loss_coef_log_dict[coef_name] = scaled_coef

    def _sum_unreduced(self, loss: torch.Tensor) -> None:
        self._loss_unreduced = [
            self._sum_tensor_aggregator(self._loss_unreduced + [loss])
        ]

    def process_structured_loss(
        self, link_name: str, structured_loss: StructuredLoss
    ) -> None:

        loss_name = structured_loss["name"]
        full_loss_name = self._format_full_loss_name(
            link_name=link_name, loss_name=loss_name
        )

        coef = structured_loss["coef"]
        scaled_coef = self._scale_loss_coef(full_loss_name=full_loss_name, coef=coef)
        self._add_scaled_loss_coef_to_loss_coef_log_dict(
            full_loss_name=full_loss_name, scaled_coef=scaled_coef
        )

        if structured_loss["reduced"]:
            self._loss_reduced += scaled_coef * structured_loss["data"]
            loss = structured_loss["data"]
        else:
            self._sum_unreduced(loss=scaled_coef * structured_loss["data"])
            loss = structured_loss["data"].mean()

        self._add_loss_to_loss_log_dict(full_loss_name=full_loss_name, loss=loss)

    def _reduce_unreduced(self) -> torch.Tensor:
        loss = self._loss_unreduced[0]

        for reduction_spec, tensor_reductor in zip(
            self._reduction_specs, self._tensor_reductors
        ):
            if reduction_spec.pre_scale is not None:
                loss = reduction_spec.pre_scale * loss
            loss = einops_reduce(
                tensor=loss,
                pattern=reduction_spec.reduce_pattern,
                reduction=tensor_reductor,
            )
            if reduction_spec.post_scale is not None:
                loss = reduction_spec.post_scale * loss
        assert (
            loss.shape == ()
        ), f"Loss shape after aggregation should be () got {loss.shape}."

        return loss

    def get_loss(self) -> torch.Tensor:
        loss: torch.Tensor = self._loss_reduced

        if len(self._loss_unreduced) != 0:
            loss += self._reduce_unreduced()

        self.reset()
        return loss

    def loss_coef_schedulers_step(self) -> None:
        # Should be called in 'on_train_epoch_end' method of the parent pytorch_lightning.LightningModule
        for loss_coef_scheduler in self._loss_coef_schedulers.values():
            loss_coef_scheduler.step()

    def _scale_loss_coef(self, full_loss_name: str, coef: float) -> float:
        loss_coef_scheduler = self._loss_coef_schedulers.get(full_loss_name, None)
        if loss_coef_scheduler is not None:
            coef_scale = loss_coef_scheduler.get_coef_scale()
            coef *= coef_scale
        return coef

    def state_dict(self) -> Dict[str, Any]:
        return {
            full_loss_name: loss_coef_scheduler.state_dict
            for full_loss_name, loss_coef_scheduler in self._loss_coef_schedulers.items()
        }

    # Load the state of the scheduler from checkpoint
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for full_loss_name, loss_coef_scheduler in self._loss_coef_schedulers.items():
            loss_coef_scheduler.load_state_dict(state_dict=state_dict[full_loss_name])

    @staticmethod
    def _format_full_loss_name(link_name: str, loss_name: str) -> str:
        return (
            loss_name + "-" + link_name
        )  # Loss name gets long and prone to different links outputting the same loss type.
