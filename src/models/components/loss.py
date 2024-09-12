r"""
This module contains various classes and functions for managing and computing different types of loss functions in a machine learning model. It includes utilities for mapping loss names, retrieving explicit constraints, and managing reconstruction losses. Additionally, it provides classes for scheduling loss coefficients over epochs and chaining multiple schedulers together.

Classes:
    ReconstructionLoss(nn.Module):

    BaseLossCoefScheduler:
        Base class for scheduling coefficient scaling over a number of epochs.

    LinearLossCoefScheduler(BaseLossCoefScheduler):
        Scheduler that linearly interpolates the loss coefficient scale over a number of epochs.

    ExponentialLossCoefScheduler(BaseLossCoefScheduler):
        Scheduler that adjusts the coefficient scale exponentially over epochs.

    ChainedLossCoefScheduler:
        Manages a sequence of loss coefficient schedulers, stepping through each scheduler sequentially.

    LossManager:
        Manages the computation, reduction, and logging of loss values in a machine learning model.

Functions:
    map_loss_name(loss_name: str) -> str:

    get_explicit_constraint(constraint_name: str) -> Callable[[torch.Tensor, int | None], torch.Tensor]:


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from typing import List, Any, Callable, Dict
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
    r"""
    Maps a given loss name to its corresponding loss name displayed during logging.

    Args:
        loss_name (str): The name of the loss function to be mapped.

    Returns:
        str: The corresponding loss function name if found.

    Raises:
        ValueError: If the provided loss_name is not found in the _LOSS_NAME_MANAGER dictionary.
    """

    loss = _LOSS_NAME_MANAGER.get(loss_name, None)

    if loss is not None:
        return loss

    raise ValueError(
        f"The provided loss_name {loss_name} is wrong. Must be one of {' ,'.join(list(_LOSS_NAME_MANAGER.keys()))}"
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
    r"""
    Retrieve an explicit constraint function based on the provided constraint name.

    Args:
        constraint_name (str): The name of the constraint to retrieve.

    Returns:
        Callable[[torch.Tensor, int | None], torch.Tensor]: The constraint function corresponding to the provided name.

    Raises:
        ValueError: If the provided constraint_name does not match any known constraints.
    """

    constraint = _EXPLICIT_CONSTRAINT.get(constraint_name, None)

    if constraint is None:
        raise ValueError(
            f"The provided constraint_name {constraint_name} is wrong. Must be one of {' ,'.join(list(_EXPLICIT_CONSTRAINT.keys()))}"
        )

    return constraint


_RECONSTRUCTION_LOSSES: Dict[
    str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
] = {
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "smooth_l1": nn.SmoothL1Loss,
}


def _get_reconstruction_loss(
    loss_name: str,
    kwargs: Dict[str, Any],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    r"""
    Retrieves a reconstruction loss function based on the provided loss name and keyword arguments.

    Args:
        loss_name (str): The name of the reconstruction loss function to retrieve.
        kwargs (Dict[str, Any]): A dictionary of keyword arguments to pass to the loss function.

    Returns:
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: The reconstruction loss function.

    Raises:
        ValueError: If the provided loss_name is not found in the available reconstruction losses.
    """

    loss = _RECONSTRUCTION_LOSSES.get(loss_name, None)

    if loss is not None:
        return loss(**kwargs)

    raise ValueError(
        f"The provided loss_name {loss_name} is wrong. Must be one of {' ,'.join(list(_RECONSTRUCTION_LOSSES.keys()))}"
    )


class ReconstructionLoss(nn.Module):
    """
    A PyTorch module for computing the reconstruction loss.

    Attributes:
        _config_structure (ConfigStructure): The expected structure of the configuration.
        _coef (float): Coefficient for the reconstruction loss.
        _loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function used for reconstruction.
        _tensor_reductor (TensorReductor): The tensor reductor used for reducing the loss tensor.

    Methods:
        __init__(cfg: Namespace) -> None:
            Initializes the ReconstructionLoss module with the given configuration.

        forward(batch: Batch, data_name: str, target_name: str, reduce_pattern: str) -> StructuredForwardOutput:
            Computes the reconstruction loss for the given batch and returns the structured forward output.
    """

    _config_structure: ConfigStructure = {
        "coef": float,
        "loss_fn": {"name": str, "kwargs": Namespace},
        "tensor_reductor": {"name": str, "kwargs": Namespace},
    }

    def __init__(self, cfg: Namespace) -> None:
        super(ReconstructionLoss, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        # Setup reconstruction loss coefficient.
        self._coef: float = cfg.coef

        # Initialize loss function.
        self._loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
            _get_reconstruction_loss(
                loss_name=cfg.loss_fn.name, kwargs=vars(cfg.loss_fn.kwargs)
            )
        )

        # Initialize tensor reductor used for reducing tensor with losses.
        self._tensor_reductor: TensorReductor = get_tensor_reductor(
            tensor_reductor_name=cfg.tensor_reductor.name,
            kwargs=vars(cfg.tensor_reductor.kwargs),
        )

    @staticmethod
    def _parse_hparams_to_dict(cfg: Namespace) -> Dict[str, Any]:
        r"""
        Parses hyperparameters from a Namespace object to a dictionary.

        Args:
            cfg (Namespace): A Namespace object containing the hyperparameters.

        Returns:
            Dict[str, Any]: A dictionary with the parsed hyperparameters.
                - "coef": The coefficient value from the cfg.
                - "loss_fn": A dictionary of attributes from the loss function in cfg.
                - "tensor_reductor": A dictionary of attributes from the tensor reductor in cfg.
        """

        return {
            "coef": cfg.coef,
            "loss_fn": vars(cfg.loss_fn),
            "tensor_reductor": vars(cfg.tensor_reductor),
        }

    def forward(
        self, batch: Batch, data_name: str, target_name: str, reduce_pattern: str
    ) -> StructuredForwardOutput:
        r"""
        Computes the forward pass for the loss calculation.

        Args:
            batch (Batch): The input batch containing data and target tensors.
            data_name (str): The key in the batch dictionary for the input data tensor.
            target_name (str): The key in the batch dictionary for the target tensor.
            reduce_pattern (str): The pattern used for reducing the loss tensor.

        Returns:
            StructuredForwardOutput: The structured output containing the computed loss.
        """

        loss: torch.Tensor = self._loss_fn(batch[data_name], batch[target_name])

        loss = einops_reduce(
            tensor=loss, pattern=reduce_pattern, reduction=self._tensor_reductor
        )

        # Check if shape is torch.Size([1]).
        reduced = loss.shape == (1,)

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
    r"""
    BaseLossCoefScheduler is a base class for scheduling coefficient scaling over a number of epochs.

    Attributes:
        _start_coef_scale (float): The initial coefficient scale.
        _end_coef_scale (float): The final coefficient scale.
        epoch_count (int): The current epoch count.
        num_epochs (int): The total number of epochs over which to schedule the coefficient scaling.

    Methods:
        __init__(start_coef_scale: float, end_coef_scale: float, num_epochs: int) -> None:
            Initializes the scheduler with the starting and ending coefficient scales and the number of epochs.

        step() -> float:
            Increases the epoch count and returns the updated coefficient scale.

        get_coef_scale() -> float:
            Abstract method that should be implemented in subclasses to calculate the coefficient scale.

        state_dict() -> dict[str, Any]:
            Returns a dictionary containing the current state of the scheduler.

        load_state_dict(state_dict: dict[str, Any]) -> None:
            Loads the state of the scheduler from a dictionary.
    """

    def __init__(
        self, start_coef_scale: float, end_coef_scale: float, num_epochs: int
    ) -> None:

        self._start_coef_scale: float = start_coef_scale
        # self._current_coef_scale: float = start_coef_scale
        self._end_coef_scale: float = end_coef_scale
        self.epoch_count: int = 0
        self.num_epochs: int = num_epochs

    def step(self) -> float:
        r"""
        Increment the epoch count, update the coefficient scale, and return the updated coefficient.

        This method increases the internal epoch counter by one, ensuring it does not exceed the
        maximum number of epochs (`num_epochs`). It then updates the current coefficient scale
        using the `get_coef_scale` method and returns the updated coefficient scale.

        Returns:
            float: The updated coefficient scale after incrementing the epoch count.
        """

        self.epoch_count += 1
        self.epoch_count = min(self.num_epochs, self.epoch_count)
        self._current_coef_scale = self.get_coef_scale()

        return self.get_coef_scale()

    def get_coef_scale(self) -> float:
        r"""This method should be implemented in subclasses."""
        raise NotImplementedError

    # THIS WOULD BE COOL TO IMPLEMENT ALONG WITH USE OF self._current_coef_scale NOT TO CALCULATE EVERYTHIN FROM SCRATCH EACH TIME AS DONE IN LR SCHEDULERS IN PYTORCH BUT I GUESS IT HAS MARGINAL SIGNIFICANCE
    # get_coef_scale is called once per batch for each loss coef scheduler calculations could be reused
    # def _get_closed_form_coef(self) -> float:
    #     """This method should be implemented in subclasses."""
    #     raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        r"""
        Save the current state.

        Returns:
            Dict[str, Any]: A dictionary containing the current state with the key 'epoch_count'.
        """

        return {"epoch_count": self.epoch_count}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        r"""
        Load the saved state from a dictionary.

        Args:
            state_dict (Dict[str, Any]): A dictionary containing the state to load.
                                         Expected keys are:
                                         - "epoch_count": The count of epochs completed.
        """

        self.epoch_count = state_dict["epoch_count"]
        # self._current_coef_scale = self._get_closed_form_coef()


class LinearLossCoefScheduler(BaseLossCoefScheduler):
    r"""
    A scheduler that linearly interpolates the loss coefficient scale over a number of epochs.

    Attributes:
        _start_coef_scale (float): The initial coefficient scale at the start of the training.
        _end_coef_scale (float): The final coefficient scale at the end of the training.
        epoch_count (int): The current epoch number.
        num_epochs (int): The total number of epochs over which to schedule the coefficient scale.

    Methods:
        get_coef_scale() -> float:
            Computes the current coefficient scale based on the linear interpolation between
            the start and end coefficient scales, proportionate to the current epoch.
    """

    def get_coef_scale(self) -> float:
        return self._start_coef_scale + (
            self._end_coef_scale - self._start_coef_scale
        ) * (self.epoch_count / self.num_epochs)


class ExponentialLossCoefScheduler(BaseLossCoefScheduler):
    r"""
    A scheduler that adjusts the coefficient scale exponentially over epochs.

    Attributes:
        _start_coef_scale (float): The initial coefficient scale.
        _end_coef_scale (float): The final coefficient scale.
        num_epochs (int): The total number of epochs over which to adjust the coefficient scale.
        epoch_count (int): The current epoch count.

    Methods:
        get_coef_scale() -> float:
            Calculates and returns the coefficient scale for the current epoch.
    """

    def get_coef_scale(self) -> float:
        decay_rate = (self._end_coef_scale / self._start_coef_scale) ** (
            1 / self.num_epochs
        )
        return self._start_coef_scale * (decay_rate**self.epoch_count)


# NOTE: The below ChainedLossCoefScheduler scheduler has different | better :) logic than ChainedScheduler from pytorch.
class ChainedLossCoefScheduler:
    r"""ChainedLossCoefScheduler

    A class to manage a sequence of loss coefficient schedulers. It steps through each scheduler sequentially,
    moving to the next scheduler once the current one is done.

    Attributes:
        schedulers (List[BaseLossCoefScheduler]): A list of scheduler instances that will be applied sequentially.
        current_scheduler_idx (int): The index of the current scheduler being used.

    Methods:
        step() -> float:
            Step through the current scheduler. If it's done, move to the next scheduler.

        get_coef_scale() -> float:
            Get the coefficient scale value from the current scheduler.

        state_dict() -> Dict[str, Any]:
            Save the state of all schedulers.

        load_state_dict(state_dict: Dict[str, Any]) -> None:
            Load the state of all schedulers.
    """

    def __init__(self, schedulers: List[BaseLossCoefScheduler]):

        self.schedulers: List[BaseLossCoefScheduler] = schedulers
        self.current_scheduler_idx: int = 0

    def step(self) -> float:
        r"""
        Step through the current scheduler. If the current scheduler has completed its epochs,
        move to the next scheduler in the list.

        Returns:
            float: The coefficient scale from the current scheduler's step.
        """

        current_scheduler = self.schedulers[self.current_scheduler_idx]
        coef_scale: float = current_scheduler.step()

        # Check if the current scheduler is done. If it is the case then do not change the current scheduler index.
        if current_scheduler.epoch_count >= current_scheduler.num_epochs:
            if self.current_scheduler_idx < len(self.schedulers) - 1:
                self.current_scheduler_idx += 1

        return coef_scale

    def get_coef_scale(self) -> float:
        r"""
        Returns:
            float: The coefficient scale value.
        """
        current_scheduler = self.schedulers[self.current_scheduler_idx]
        coef_scale = current_scheduler.get_coef_scale()
        return coef_scale

    def state_dict(self) -> Dict[str, Any]:
        r"""
        Save the state of all schedulers.

        Returns:
            Dict[str, Any]: A dictionary containing the state of all schedulers and the current scheduler index.
        """

        return {
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
            "current_scheduler_idx": self.current_scheduler_idx,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        r"""
        Load the state of the schedulers from a given state dictionary.

        Args:
            state_dict (Dict[str, Any]): A dictionary containing the state of the schedulers.
                - "schedulers" (List[Dict[str, Any]]): A list of state dictionaries for each scheduler.
                - "current_scheduler_idx" (int): The index of the current scheduler.
        """

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
    r"""
    Retrieve and instantiate a loss coefficient scheduler based on the provided scheduler name.

    Args:
        scheduler_name (str): The name of the scheduler to retrieve.
        kwargs (Dict[str, Any]): A dictionary of keyword arguments to pass to the scheduler's constructor.

    Returns:
        BaseLossCoefScheduler: An instance of the requested loss coefficient scheduler.

    Raises:
        ValueError: If the provided scheduler_name is not found in the _LOSS_COEF_SCHEDULERS dictionary.
    """

    coef_scheduler_class = _LOSS_COEF_SCHEDULERS.get(scheduler_name, None)

    if coef_scheduler_class is None:
        raise ValueError(
            f"The provided scheduler_name {coef_scheduler_class} is invalid. Must be one of {list(_LOSS_COEF_SCHEDULERS.keys())}"
        )

    return coef_scheduler_class(**kwargs)


def _get_chained_loss_coef_scheduler(
    schedulers_cfg: List[Namespace],
) -> ChainedLossCoefScheduler:
    r"""
    Creates a ChainedLossCoefScheduler from a list of scheduler configurations.

    Args:
        schedulers_cfg (List[Namespace]): A list of Namespace objects, each containing
                                          the configuration for a loss coefficient scheduler.

    Returns:
        ChainedLossCoefScheduler: An instance of ChainedLossCoefScheduler initialized
                                  with the specified schedulers.
    """

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
    r"""LossManager is a class that manages the computation, reduction, and logging of loss values in a machine learning model.

    Attributes:
        _config_structure (ConfigStructure): The structure of the configuration dictionary.
        _sum_tensor_aggregator (TensorAggregator): Function used for summing up the loss terms before reducing them to a scalar.
        _reduction_specs (List[Namespace]): Specifications for reducing the summed loss terms to a scalar.
        _tensor_reductors (Callable[[torch.Tensor], torch.Tensor]): List of tensor reductor functions.
        _loss_unreduced (List[torch.Tensor]): List to store unreduced loss values.
        _loss_reduced (torch.Tensor): Tensor to store the reduced loss value, initialized to zero.
        _loss_log_dict (Dict[str, float]): Dictionary to store loss logs.
        _loss_coef_log_dict (Dict[str, float]): Dictionary to store loss coefficient logs.
        _loss_coef_schedulers (Dict[str, ChainedLossCoefScheduler]): Dictionary to store loss coefficient schedulers.

    Methods:
        __init__(cfg: Namespace) -> None:
            Initializes the LossManager with the given configuration.

        loss_log_dict -> Dict[str, float]:
            Returns the loss log dictionary.

        loss_coef_log_dict -> Dict[str, float]:
            Returns the loss coefficient log dictionary.

        _setup_reduce_loss_tensor() -> None:
            Initializes and appends tensor reductor objects to the _tensor_reductors list based on the specifications provided in _reduction_specs.

        reset() -> None:

        _add_loss_to_loss_log_dict(full_loss_name: str, loss: torch.Tensor) -> None:

        _add_scaled_loss_coef_to_loss_coef_log_dict(full_loss_name: str, scaled_coef: float) -> None:

        _sum_unreduced(loss: torch.Tensor) -> None:

        process_structured_loss(link_name: str, structured_loss: StructuredLoss) -> None:
            Processes a structured loss by scaling its coefficient, updating logs, and aggregating the loss value.

        _reduce_unreduced() -> torch.Tensor:

        get_loss() -> torch.Tensor:

        loss_coef_schedulers_step() -> None:
            Updates the loss coefficient for each loss coefficient scheduler. Should be called in 'on_train_epoch_end' method of the parent pytorch_lightning.LightningModule.

        _scale_loss_coef(full_loss_name: str, coef: float) -> float:

        state_dict() -> Dict[str, Any]:

        load_state_dict(state_dict: Dict[str, Any]) -> None:

    Static Methods:
        _format_full_loss_name(link_name: str, loss_name: str) -> str:


    """

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

    @staticmethod
    def _parse_hparams_to_dict(self, cfg: Namespace) -> Dict[str, Any]:
        return cfg

    @property
    def loss_log_dict(self) -> Dict[str, float]:
        return self._loss_log_dict

    @property
    def loss_coef_log_dict(self) -> Dict[str, float]:
        return self._loss_coef_log_dict

    def _setup_reduce_loss_tensor(self) -> None:
        r"""
        Initializes and appends tensor reductor objects to the _tensor_reductors list
        based on the specifications provided in _reduction_specs.

        Each reductor is created using the get_tensor_reductor function, which takes
        the reductor name and keyword arguments from the reduction_spec object.

        Returns:
            None
        """

        for reduction_spec in self._reduction_specs:
            self._tensor_reductors.append(
                get_tensor_reductor(
                    tensor_reductor_name=reduction_spec.reductor_name,
                    kwargs=vars(reduction_spec.kwargs),
                )
            )

    def reset(self) -> None:
        r"""
        Resets the internal state of the loss component.

        This method clears the lists and dictionaries used to store loss values
        and logs, and initializes the reduced loss tensor to zero.

        Attributes reset:
            _loss_unreduced (list): A list to store unreduced loss values.
            _loss_reduced (torch.Tensor): A tensor to store the reduced loss value, initialized to zero.
            _loss_log_dict (dict): A dictionary to store loss logs.
            _loss_coef_log_dict (dict): A dictionary to store loss coefficient logs.
        """

        self._loss_unreduced = []
        self._loss_reduced = torch.zeros(1, dtype=torch.float32)
        self._loss_log_dict = {}
        self._loss_coef_log_dict = {}

    def _add_loss_to_loss_log_dict(
        self, full_loss_name: str, loss: torch.Tensor
    ) -> None:
        r"""
        Adds a loss value to the loss log dictionary with a unique name.

        Args:
            full_loss_name (str): The full name of the loss to be added.
            loss (torch.Tensor): The loss value as a tensor.

        Raises:
            AssertionError: If the full_loss_name contains a '.' character.

        Notes:
            The method ensures that the loss name is unique within the loss log dictionary.
            If a loss name already exists, it appends a numeric suffix to create a unique name.
        """

        assert (
            "." not in full_loss_name
        ), f"Character '.' cannot occur in the link_name as it is used in logic for disemabiguating link loss calculation number."

        loss_log_name = full_loss_name

        # If the link is runned multiple times the loss name is extended.
        if loss_log_name in self._loss_log_dict:
            loss_log_name += ".1"
            while loss_log_name in self._loss_log_dict:
                splitted = loss_log_name.split(".")
                loss_log_name = splitted(-2) + "." + str(int(splitted(-1)) + 1)

        # Save the reducted loss to the loss log dict returned for logging.
        self._loss_log_dict[loss_log_name] = loss.item()

    def _add_scaled_loss_coef_to_loss_coef_log_dict(
        self, full_loss_name: str, scaled_coef: float
    ) -> None:
        r"""
        Adds a scaled loss coefficient to the loss coefficient log dictionary.

        This method constructs a coefficient name by appending "_loss_coef" to the
        provided full loss name and then adds the scaled coefficient to the
        `_loss_coef_log_dict` with the constructed name as the key.

        Args:
            full_loss_name (str): The full name of the loss function.
            scaled_coef (float): The scaled coefficient to be added to the log dictionary.

        Returns:
            None
        """

        coef_name = full_loss_name + "_loss_coef"
        self._loss_coef_log_dict[coef_name] = scaled_coef

    def _sum_unreduced(self, loss: torch.Tensor) -> None:
        r"""
        Aggregates the given loss tensor into the internal list of unreduced losses.

        Args:
            loss (torch.Tensor): The loss tensor to be added to the list of unreduced losses.

        Returns:
            None
        """

        self._loss_unreduced = [
            self._sum_tensor_aggregator(self._loss_unreduced + [loss])
        ]

    def process_structured_loss(
        self, link_name: str, structured_loss: StructuredLoss
    ) -> None:
        r"""
        Processes a structured loss by scaling its coefficient, updating logs, and
        aggregating the loss value.

        Args:
            link_name (str): The name of the link associated with the loss.
            structured_loss (StructuredLoss): A dictionary containing the loss details
                                              with keys "name", "coef", "reduced", and "data".

        Returns:
            None
        """

        loss_name = structured_loss["name"]
        full_loss_name = self._format_full_loss_name(
            link_name=link_name, loss_name=loss_name
        )

        coef = structured_loss["coef"]

        # Scale loss coef with the loss coef scheduler.
        scaled_coef = self._scale_loss_coef(full_loss_name=full_loss_name, coef=coef)
        self._add_scaled_loss_coef_to_loss_coef_log_dict(
            full_loss_name=full_loss_name, scaled_coef=scaled_coef
        )

        # Process differently reduced and unreduced losses.
        if structured_loss["reduced"]:
            self._loss_reduced += scaled_coef * structured_loss["data"]
            loss = structured_loss["data"]
        else:
            self._sum_unreduced(loss=scaled_coef * structured_loss["data"])
            loss = structured_loss["data"].mean()

        # Add the loss to logging dict.
        self._add_loss_to_loss_log_dict(full_loss_name=full_loss_name, loss=loss)

    def _reduce_unreduced(self) -> torch.Tensor:
        r"""
        Reduces the unreduced loss tensor according to specified reduction specifications.

        This method iterates over the reduction specifications and tensor reductors,
        applying pre-scaling, reduction, and post-scaling operations to the loss tensor.

        Returns:
            torch.Tensor: The reduced loss tensor, which should be a scalar.

        Raises:
            AssertionError: If the final reduced loss tensor is not a scalar.
        """

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
        r"""
        Computes and returns the loss tensor.

        This method calculates the loss by first taking the reduced loss tensor.
        If there are any unreduced loss values, it adds the reduced version of those
        to the total loss. After computing the total loss, it resets the internal state.

        Returns:
            torch.Tensor: The computed loss tensor.
        """

        loss: torch.Tensor = self._loss_reduced

        if len(self._loss_unreduced) != 0:
            loss += self._reduce_unreduced()

        self.reset()
        return loss

    def loss_coef_schedulers_step(self) -> None:
        r"""
        Updates the loss coefficient for each loss coefficient scheduler.
        Should be called in 'on_train_epoch_end' method of the parent pytorch_lightning.LightningModule.

        This method should be called in the 'on_train_epoch_end' method of the
        parent `pytorch_lightning.LightningModule`. It iterates through all
        loss coefficient schedulers stored in `self._loss_coef_schedulers`
        and calls their `step` method to update the loss coefficient.

        Returns:
            None
        """

        for loss_coef_scheduler in self._loss_coef_schedulers.values():
            loss_coef_scheduler.step()

    def _scale_loss_coef(self, full_loss_name: str, coef: float) -> float:
        r"""
        Scales the loss coefficient based on a scheduler if available.

        Args:
            full_loss_name (str): The full name of the loss function.
            coef (float): The initial coefficient value.

        Returns:
            float: The scaled coefficient value.
        """

        scaled_coef: float = coef
        loss_coef_scheduler = self._loss_coef_schedulers.get(full_loss_name, None)

        if loss_coef_scheduler is not None:
            coef_scale = loss_coef_scheduler.get_coef_scale()
            scaled_coef *= coef_scale

        return scaled_coef

    def state_dict(self) -> Dict[str, Any]:
        r"""
        Returns the state dictionary of the loss coefficient schedulers.

        This method creates and returns a dictionary where the keys are the names
        of the loss functions and the values are the state dictionaries of their
        corresponding coefficient schedulers.

        Returns:
            Dict[str, Any]: A dictionary containing the state dictionaries of the
            loss coefficient schedulers.
        """

        return {
            full_loss_name: loss_coef_scheduler.state_dict()
            for full_loss_name, loss_coef_scheduler in self._loss_coef_schedulers.items()
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        r"""
        Load the state dictionary for the loss coefficient schedulers.

        Args:
            state_dict (Dict[str, Any]): A dictionary containing the state of each loss coefficient scheduler.
                                         The keys should correspond to the full names of the loss coefficient schedulers,
                                         and the values should be the state dictionaries for each scheduler.

        Returns:
            None
        """

        for full_loss_name, loss_coef_scheduler in self._loss_coef_schedulers.items():
            loss_coef_scheduler.load_state_dict(state_dict=state_dict[full_loss_name])

    @staticmethod
    def _format_full_loss_name(link_name: str, loss_name: str) -> str:
        r"""
        Formats the full loss name by combining the loss name and link name.

        This function concatenates the loss name and link name with a hyphen ("-")
        to create a full loss name. This helps in distinguishing between different
        links that might output the same type of loss.

        Args:
            link_name (str): The name of the link.
            loss_name (str): The name of the loss.

        Returns:
            str: The formatted full loss name.
        """

        return (
            loss_name + "-" + link_name
        )  # Loss name gets long and prone to different links outputting the same loss type.
