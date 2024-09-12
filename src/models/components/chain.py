r"""
A module containing logic of the model creation from configuration.

This module defines Chain i.e. a highly configurable pl.LightningModule subclass build of so ChainLink i.e.
nn.Module subclassed implemented in other project modules and referenced in _CHAIN_LINKS dict.

Chain allows to specify a wild wild range of models via simple yet powerful .yaml configuration file (just 200 lines :) -> Don't let others be quicker; switch your programming language from Python to YAML!
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import inspect
from argparse import Namespace
from typing import List, Dict, TypeAlias, Type, Any

from utils.config import validate_config_structure, parse_choice_spec_path
from utils.common_types import Batch, ConfigStructure
from models.components.dense_blocks import DenseBlockStack, StandaloneTinyNNModule
from models.components.latent import (
    GaussianPosterior,
    GaussianMixturePriorNLL,
    LatentConstraint,
    FuzzyClustering,
    VectorConditionedLogitsGMPriorNLL,
)
from models.components.misc import (
    AggregateDataAdapter,
    BatchRearranger,
    TensorCloner,
    BatchRepeater,
)
from models.components.condition_embedding import ConditionSetEmbeddingTransformer
from models.components.loss import LossManager, ReconstructionLoss
from models.components.optimizer import get_chained_scheduler, get_optimizer

ChainLink: TypeAlias = Type[nn.Module]

_DENSE_BLOCKS_MODULE: Dict[str, ChainLink] = {
    "dense_block_stack": DenseBlockStack,
    "standalone_tiny_module": StandaloneTinyNNModule,
}

_LATENT_MODULE: Dict[str, ChainLink] = {
    "posterior_gaussian": GaussianPosterior,
    "prior_gaussian_mixture_nll": GaussianMixturePriorNLL,
    "constraint_latent": LatentConstraint,
    "clustering_fuzzy": FuzzyClustering,
    "prior_vec_conditioned_logits_gaussian_mixture_nll": VectorConditionedLogitsGMPriorNLL,
}

_MISC_MODULE: Dict[str, ChainLink] = {
    "adapter": AggregateDataAdapter,
    "rearranger": BatchRearranger,
    "cloner": TensorCloner,
    "repeater": BatchRepeater,
}

_CONDITION_EMBEDDING_MODULE: Dict[str, ChainLink] = {
    "condition_set_embedding_transformer": ConditionSetEmbeddingTransformer
}

_LOSS_MODULE: Dict[str, ChainLink] = {"reconstruction": ReconstructionLoss}

_CHAIN_LINKS: Dict[str, Dict[str, ChainLink]] = {
    "dense_blocks": _DENSE_BLOCKS_MODULE,
    "latent": _LATENT_MODULE,
    "misc": _MISC_MODULE,
    "condition_embedding": _CONDITION_EMBEDDING_MODULE,
    "loss": _LOSS_MODULE,
}


class ChainLinkSpec(Namespace):
    r"""
    Represents the specification of a ChainLink object i.e. subclass of nn.Module.

    Attributes:
        chain_link_spec_path (str): The path of the chain link specification in the _CHAIN_LINKS dict.
        cfg (Namespace): The configuration of the chain link.
    """

    chain_link_spec_path: str
    cfg: Namespace


def _get_chain_link(chain_link_spec: Namespace) -> ChainLink:
    r"""
    Retrieves a ChainLink object based on the provided chain link specification.

    Args:
        chain_link_spec (Namespace): The chain link specification.

    Returns:
        ChainLink: The retrieved chain link.

    Raises:
        ValueError: If the provided key is wrong and not found in the chain link specification.
    """

    choice_path: List[str] = parse_choice_spec_path(
        spec_path=chain_link_spec.chain_link_spec_path
    )

    choice = _CHAIN_LINKS
    for key in choice_path:
        next_choice = choice.get(key, None)
        if next_choice is None:
            raise ValueError(
                f"The provided key '{key}' is wrong. Must be one of {', '.join(list(choice.keys()))}"
            )
        choice = next_choice

    return choice(cfg=chain_link_spec.cfg)


class Chain(pl.LightningModule):
    r"""
    Chain is a PyTorch Lightning module that represents a chain of neural network components - ChainLink objects being nn.Module children.

    Args:
        cfg (Namespace): The configuration object containing the settings for the chain.

    Attributes:
        _config_structure (ConfigStructure): The structure of the configuration for the chain.
        automatic_optimization (bool): Flag indicating whether automatic optimization is enabled or disabled.
        _cfg (Namespace): The configuration object for the chain.
        _chain (nn.ModuleDict): The dictionary containing the chain of neural network components - ChainLinks.
        _hparams_parsed_to_dict (Dict[str, Dict | str, int]): The dictionary containing the parsed hyperparameters to be logged.
        _optimizers_cfg (List[Namespace]): The list of optimizer configurations for the chain.
        _loss_manager (LossManager): The LossManager object used for managing loss related computations.
        _commands (Dict[str, Dict[str, Dict | int]]): The dictionary containing the commands indicating computations done by the specified chain links.

    Properties:
        parsed_hparams (Dict[str, Any]): The parsed hyperparameters used for logging.

    Methods:
        _setup_chain(chain_cfg: List[Namespace]) -> None: Sets up the chain based on the configuration.
        _parse_hparams_to_dict(cfg: List[Namespace]) -> None: Parses the hyperparameters to a dictionary.
        _assert_optimizers_cfg() -> None: Asserts the correctness of the optimizer configurations.
        _parse_command_definitions(commands_definitions: List[Namespace]) -> None: Parses the command definitions.
        _assert_commands() -> None: Asserts the correctness of the commands.
        run_command(batch: Batch, command_name: str, dynamic_kwargs: Dict[str, Namespace] = {}, reset_loss_manager: bool = False) -> Batch: Runs a processing command on the batch.
        forward(batch: Batch) -> Batch: Performs the forward pass of the chain.
        training_step(batch: Batch) -> torch.Tensor: Performs a training step of the chain.
        validation_step(batch: Batch) -> torch.Tensor: Performs a validation step of the chain.
        configure_optimizers() -> dict: Configures the optimizers for the chain.
        on_train_epoch_start() -> None: Performs actions at the start of a training epoch.
        on_save_checkpoint(checkpoint: Dict[str, Any]) -> None: Performs actions when saving a checkpoint.
        on_load_checkpoint(checkpoint: Dict[str, Any]) -> None: Performs actions when loading a checkpoint.
    """

    _config_structure: ConfigStructure = {
        "chain": [{"chain_link_spec_path": str, "name": str, "cfg": Namespace}],
        "optimizers": [
            {
                "links": [str],
                "optimizer_name": str,
                "kwargs": Namespace,
                "lr_scheduler": (
                    None,
                    {
                        "schedulers": [{"name": str, "kwargs": Namespace}],
                        "out_cfg_kwargs": Namespace,
                    },
                ),
            }
        ],
        "training": {
            "batch_size": int,
            "max_epochs": int,
            "check_val_every_n_epoch": (None, int),
        },
        "loss_manager": Namespace,
        "commands": [
            {
                "command_name": str,
                "processing_steps": [{"link": str, "method": str, "kwargs": Namespace}],
            }
        ],
    }

    def __init__(self, cfg: Namespace) -> None:
        super(Chain, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self.automatic_optimization = False  # Disable automatic optimization.

        self._cfg: Namespace = cfg

        # Building chain based on the config.
        self._chain: nn.ModuleDict = nn.ModuleDict()
        self._hparams_parsed_to_dict: Dict[str, Dict | str, int] = {}
        self._setup_chain(chain_cfg=cfg.chain)

        # Setup optimizers configuration.
        self._optimizers_cfg: List[Namespace] = cfg.optimizers
        self._assert_optimizers_cfg()

        # Initializing LossMangager managing loss related computations.
        self._loss_manager: LossManager = LossManager(cfg=cfg.loss_manager)

        # Preparing and setting the commands allowing to process the batch in a specified manner.
        self._commands: Dict[str, Dict[str, Dict | int]] = {}
        self._parse_command_definitions(commands=cfg.commands)
        self._assert_commands()

        # Parsing hparams to be logged.
        self._parse_hparams_to_dict(cfg=cfg)

    def _setup_chain(self, chain_cfg: List[Namespace]) -> None:
        r"""
        Set up the self._chain based on the given chain configuration.

        Args:
            chain_cfg (List[Namespace]): The chain configuration specifying the links.

        Raises:
            AssertionError: If a link with the same name already exists in the chain.

        Returns:
            None
        """

        for link_spec in chain_cfg:
            assert (
                link_spec.name not in self._chain
            ), f"The keyword {link_spec.name} is already present in self._chain."

            self._chain[link_spec.name] = _get_chain_link(chain_link_spec=link_spec)

    def _parse_hparams_to_dict(self, cfg: List[Namespace]) -> None:
        r"""
        Parses the hyperparameters from the configuration and stores them in a dictionary.

        Args:
            cfg (List[Namespace]): The configuration containing the hyperparameters.

        Returns:
            None
        """

        for link_spec in cfg.chain:
            chain_link = self._chain[link_spec.name]
            _parse_hparams_to_dict = getattr(chain_link, "_parse_hparams_to_dict", None)

            if _parse_hparams_to_dict is not None:
                assert inspect.ismethod(_parse_hparams_to_dict) or inspect.isfunction(
                    _parse_hparams_to_dict
                ), f"Attribute '_parse_hparams_to_dict' of link '{link_spec.name}' is not a method."

                self._hparams_parsed_to_dict[link_spec.name] = _parse_hparams_to_dict(
                    cfg=link_spec.cfg
                )

        self._hparams_parsed_to_dict["optimizers"] = {}
        for optimizer_spec in cfg.optimizers:
            for link_name in optimizer_spec.links:
                self._hparams_parsed_to_dict["optimizers"][link_name] = {
                    "optimizer_name": optimizer_spec.optimizer_name,
                    "kwargs": vars(optimizer_spec.kwargs),
                    "lr_scheduler": str(optimizer_spec.lr_scheduler),
                }

        self._hparams_parsed_to_dict["loss_manager"] = (
            self._loss_manager._parse_hparams_to_dict(cfg=cfg.loss_manager)
        )

    @property
    def parsed_hparams(self) -> Dict[str, Any]:
        return {"chain_hparams": self._hparams_parsed_to_dict}

    def _assert_optimizers_cfg(self) -> None:
        r"""
        Asserts the validity of the optimizers configuration.

        Raises:
            AssertionError: If a chain link is not found in the chain.
            AssertionError: If a chain link is not assigned to any optimizer.
            AssertionError: If a chain link is assigned to more than one optimizer.
        """

        optimizer_assigned: Dict[str, bool] = {}

        for chain_link_name, chain_link in self._chain.items():
            if any(p.requires_grad for p in chain_link.parameters()):
                optimizer_assigned[chain_link_name] = False

        for optimizer_cfg in self._optimizers_cfg:
            for chain_link_name in optimizer_cfg.links:
                assert (
                    chain_link_name in self._chain
                ), f"There is  no link named '{chain_link_name}'."

                assigned = optimizer_assigned.get(chain_link_name, None)

                assert (
                    assigned is not None
                ), f"The chain link '{chain_link_name}' is not assigned to any optimizer."
                assert (
                    not assigned
                ), f"The chain link '{chain_link_name}' cannot be assigned to more than one optimizer."

                optimizer_assigned[chain_link_name] = True

    def _parse_command_definitions(self, commands: List[Namespace]) -> None:
        r"""
        Parses the command definitions and stores them in the `_commands` dictionary.

        Args:
            commands (List[Namespace]): A list of command objects containing the command name and processing steps.

        Returns:
            None
        """

        for command in commands:
            self._commands[command.command_name] = command.processing_steps

    def _assert_commands(self) -> None:
        r"""
        Asserts the validity of commands in the chain.

        Raises:
            AssertionError: If the 'forward' command is not present in self._commands.
            AssertionError: If any command processing step's link does not match any link name in self._chain.
            AssertionError: If any link in self._chain does not have the specified method.
            AssertionError: If the specified method in a link is not a method or function.
        """

        assert (
            "forward" in self._commands
        ), "There must be a command named forward among the self._commands."

        for command_key, command_processing_steps in self._commands.items():
            for cmd_proc_step in command_processing_steps:
                assert (
                    cmd_proc_step.link in self._chain
                ), f"The '{cmd_proc_step.link}' does not match with any link name from _chain {list(self._chain.keys())}."

                link = self._chain[cmd_proc_step.link]
                method = getattr(link, cmd_proc_step.method, None)

                assert (
                    method is not None
                ), f"Link '{cmd_proc_step.link}' has no attribute named {cmd_proc_step.method}."
                assert inspect.ismethod(method) or inspect.isfunction(
                    method
                ), f"Attribute '{cmd_proc_step.method}' of link '{cmd_proc_step.link}' is not a method."

    def run_command(
        self,
        batch: Batch,
        command_name: str,
        dynamic_kwargs: Dict[str, Namespace] = {},
        reset_loss_manager: bool = False,
    ) -> Batch:
        r"""
        Executes a command on the chain.

        Args:
            batch (Batch): The input batch.
            command_name (str): The name of the command to be executed.
            dynamic_kwargs (Dict[str, Namespace], optional): Dynamic keyword arguments for the command added when calling a command. Defaults to {}.
            reset_loss_manager (bool, optional): Whether to reset the loss manager. Defaults to False.

        Returns:
            Batch: The modified batch after executing the command.
        """

        assert (
            command_name in self._commands
        ), f"The provivided command {command_name} is not within defined commands {list(self._commands.keys())}."

        # Checking dynamic_kwargs to be passed to links methods to be called.
        for link_name, kwargs in dynamic_kwargs.items():
            assert (
                link_name in self._chain
            ), f"The link of name '{link_name}' is not present in self._chain."
            assert isinstance(
                kwargs, Namespace
            ), "The provided kwargs are not an instance of Namespace."

        command_processing_steps = self._commands[command_name]

        # Running a specified method of each link in a specified order.
        for processing_step in command_processing_steps:
            # Getting the chain link at current processing step.
            chain_link = self._chain[processing_step.link]

            # Getting a specified method of the current chain link.
            method = getattr(chain_link, processing_step.method)

            # Getting dynamic kwargs corresponding to current chain link.
            dynamic_kwargs_step = dynamic_kwargs.get(processing_step.link, Namespace())

            # Processing the Batch object with the chain link method.
            chain_link_output = method(
                batch=batch,
                **vars(processing_step.kwargs),
                **vars(dynamic_kwargs_step),
            )
            batch = chain_link_output["batch"]

            # Passing losses to loss manager to processed.
            for structured_loss in chain_link_output["losses"]:
                self._loss_manager.process_structured_loss(
                    link_name=processing_step.link, structured_loss=structured_loss
                )

        # Resetting loss mangager after running a command if the loss should not be accumulated.
        if reset_loss_manager:
            self._loss_manager.reset()

        return batch

    def forward(self, batch: Batch) -> Batch:
        r"""
        Perform the forward pass of the chain model.

        Args:
            batch (Batch): The input batch.

        Returns:
            Batch: The output batch after the forward pass.
        """

        batch = self.run_command(
            batch=batch, command_name="forward", reset_loss_manager=False
        )

        return batch

    def training_step(self, batch: Batch) -> torch.Tensor:
        r"""
        Performs a single training step on the given batch.

        Args:
            batch (Batch): The input batch for training.

        Returns:
            torch.Tensor: The computed loss value.

        """

        # Perform forward pass.
        batch = self.forward(batch=batch)

        # Log the losses.
        self.log_dict(
            dictionary=self._loss_manager.loss_log_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )

        # Log the loss component coefficients.
        self.log_dict(
            dictionary=self._loss_manager.loss_coef_log_dict,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        # Get reduced loss.
        loss = self._loss_manager.get_loss()

        # Perform manual backward and update the parameters using every optimizer.
        self.manual_backward(loss)

        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()

        return loss

    def validation_step(self, batch: Batch) -> torch.Tensor:
        r"""
        Perform a validation step on a batch of data.

        Args:
            batch (Batch): The input batch of data.

        Returns:
            torch.Tensor: The computed loss value.

        """

        # Perform forward pass.
        batch = self.forward(batch=batch)

        # Log the losses.
        self.log_dict(
            dictionary=self._loss_manager.loss_log_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        # Get reduced loss.
        loss = self._loss_manager.get_loss()

        return loss

    def configure_optimizers(self) -> dict:
        r"""
        Configures the optimizers and learning rate schedulers for the model. See configure_optimizers docs for details.

        Returns:
            A dictionary containing the configured optimizers and their corresponding learning rate schedulers.
        """

        optimizers = []
        lr_scheduler_config_list = []

        # Go through each optimizer config and initialize corresponding optimizer.
        for optimizer_cfg in self._optimizers_cfg:
            # Getting parameters of each chain_link assigned to the optimizer. Generator of is enough as the input to the optimizer.
            params = (
                param
                for chain_link_name in optimizer_cfg.links
                for param in self._chain[chain_link_name].parameters(recurse=True)
            )
            # Setting up an optimizer according to specifications and adding it to outputted optimizers list.
            optimizer = get_optimizer(
                optimizer_name=optimizer_cfg.optimizer_name,
                params=params,
                kwargs=vars(optimizer_cfg.kwargs),
            )
            optimizers.append(optimizer)

            # If the optimizer has a learning rate scheduler assigned initialize it and create its config to be outputted.
            if optimizer_cfg.lr_scheduler is not None:
                lr_scheduler = get_chained_scheduler(
                    optimizer=optimizer,
                    schedulers_cfg=optimizer_cfg.lr_scheduler.schedulers,
                )
                # Setting lr_scheduler_cfg variable. See configure_optimizers docs for details.
                lr_scheduler_cfg = vars(optimizer_cfg.lr_scheduler.out_cfg_kwargs)
                lr_scheduler_cfg["scheduler"] = lr_scheduler

                # Add lr_scheduler to the outputted lr_scheduler_config_list. See configure_optimizers docs for details.
                lr_scheduler_config_list.append(lr_scheduler_cfg)

        return optimizers, lr_scheduler_config_list

    def on_train_epoch_start(self) -> None:
        r"""
        Function called at the beginning of each training epoch.

        This function updates the learning rate schedulers and loss coefficient schedulers.

        Parameters:
            None

        Returns:
            None
        """

        # Update learning rate schedulers
        lr_schedulers = self.lr_schedulers()
        if lr_schedulers is not None:
            if not isinstance(lr_schedulers, list):
                lr_schedulers = [lr_schedulers]

            for lr_scheduler in lr_schedulers:
                lr_scheduler.step()

        # Update loss coef schedulers
        self._loss_manager.loss_coef_schedulers_step()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        r"""
        Callback function called when saving a checkpoint during training.

        Args:
            checkpoint (Dict[str, Any]): The checkpoint dictionary to be saved.

        Returns:
            None
        """

        # Saving loss manager state dict.
        checkpoint["loss_manager"] = self._loss_manager.state_dict()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        r"""
        Loads the checkpoint state into the model.

        Args:
            checkpoint (Dict[str, Any]): The checkpoint dictionary containing the model state.

        Returns:
            None
        """

        loss_manager_state_dict = checkpoint["loss_manager"]
        self._loss_manager.load_state_dict(state_dict=loss_manager_state_dict)
