from calendar import c
from re import L
from typing import List, Dict, TypeAlias, Type, TypedDict, Callable, Any
import torch
import torch.nn as nn
import pytorch_lightning as pl
import inspect
from argparse import Namespace

from utils.config import validate_config_structure, parse_choice_spec_path
from utils.common_types import Batch, ConfigStructure
from models.components.blocks import BlockStack, StandaloneTinyModule
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
from models.components.condition_embedding import ConditionEmbeddingTransformer
from models.components.loss import LossManager, ReconstructionLoss
from models.components.optimizer import get_chained_scheduler, get_optimizer

ChainLink: TypeAlias = Type[nn.Module]

_BLOCKS_MODULE: Dict[str, ChainLink] = {
    "block_stack": BlockStack,
    "standalone_tiny_module": StandaloneTinyModule,
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
    "condition_embedding_transformer": ConditionEmbeddingTransformer
}

_LOSS_MODULE: Dict[str, ChainLink] = {"reconstruction": ReconstructionLoss}

_CHAIN_LINKS: Dict[str, Dict[str, ChainLink]] = {
    "blocks": _BLOCKS_MODULE,
    "latent": _LATENT_MODULE,
    "misc": _MISC_MODULE,
    "condition_embedding": _CONDITION_EMBEDDING_MODULE,
    "loss": _LOSS_MODULE,
}


class ChainLinkSpec(Namespace):
    chain_link_spec_path: str
    cfg: Namespace


def _get_chain_link(chain_link_spec: Namespace) -> ChainLink:
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
        "loss_manager": Namespace,
        "processing_commands": [
            {
                "command_name": str,
                "processing_steps": [{"link": str, "method": str, "kwargs": Namespace}],
            }
        ],
    }

    def __init__(self, cfg: Namespace) -> None:
        super(Chain, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self.automatic_optimization = False  # Disable automatic optimization

        self._cfg: Namespace = cfg

        # Building chain based on the config.
        self._chain: nn.ModuleDict = nn.ModuleDict()
        self._hparams_parsed_to_dict: Dict[str, Dict | str, int] = {}
        self._setup_chain(chain_cfg=cfg.chain)

        # Setup optimizers config.
        self._optimizers_cfg: List[Namespace] = cfg.optimizers
        self._assert_optimizers_cfg()

        # Initializing LossMangager.
        self._loss_manager: LossManager = LossManager(cfg=cfg.loss_manager)

        # Preparing and setting the processing commands allowing to process the batch in a flexible manner.
        self._processing_commands: Dict[str, Dict[str, Dict | int]] = {}
        self._parse_processing_command_definitions(
            processing_commands=cfg.processing_commands
        )
        self._assert_commands()

        # Parsing hparams
        self._parse_hparams_to_dict(cfg=cfg)

    def _setup_chain(self, chain_cfg: List[Namespace]) -> None:
        for link_spec in chain_cfg:
            assert (
                link_spec.name not in self._chain
            ), f"The keyword {link_spec.name} is already present in self._chain."
            self._chain[link_spec.name] = _get_chain_link(chain_link_spec=link_spec)

    def _parse_hparams_to_dict(self, cfg: List[Namespace]) -> None:
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

    @property
    def parsed_hparams(self) -> Dict[str, Any]:
        return {"chain_hparams": self._hparams_parsed_to_dict}

    def _assert_optimizers_cfg(self) -> None:
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

    def _parse_processing_command_definitions(
        self, processing_commands: List[Namespace]
    ) -> None:
        for command in processing_commands:
            self._processing_commands[command.command_name] = command.processing_steps

    def _assert_commands(self) -> None:
        assert (
            "forward" in self._processing_commands
        ), "There must be a command named forward among the self._processing_commands."
        for command_key, command_processing_steps in self._processing_commands.items():
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

    def run_processing_command(
        self,
        batch: Batch,
        command_name: str,
        dynamic_kwargs: Dict[str, Namespace] = {},
        reset_loss_manager: bool = False,
    ) -> Batch:
        assert (
            command_name in self._processing_commands
        ), f"The provivided command {command_name} is not within defined commands {list(self._processing_commands.keys())}."
        for link_name, kwargs in dynamic_kwargs.items():
            assert (
                link_name in self._chain
            ), f"The link of name '{link_name}' is not present in self._chain."
            assert isinstance(
                kwargs, Namespace
            ), "The provided kwargs are not an instance of Namespace."

        command_processing_steps = self._processing_commands[command_name]

        # Processing
        for processing_step in command_processing_steps:
            chain_link = self._chain[processing_step.link]
            method = getattr(chain_link, processing_step.method)
            dynamic_kwargs_step = dynamic_kwargs.get(processing_step.link, Namespace())
            chain_link_output = method(
                batch=batch,
                **vars(processing_step.kwargs),
                **vars(dynamic_kwargs_step),
            )
            batch = chain_link_output["batch"]
            for structured_loss in chain_link_output["losses"]:
                self._loss_manager.process_structured_loss(
                    link_name=processing_step.link, structured_loss=structured_loss
                )

        if reset_loss_manager:
            self._loss_manager.reset()

        return batch

    def forward(self, batch: Batch) -> Batch:
        batch = self.run_processing_command(batch=batch, command_name="forward")

        return batch

    def training_step(self, batch: Batch) -> torch.Tensor:

        batch = self.forward(batch=batch)

        self.log_dict(
            dictionary=self._loss_manager.loss_log_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )

        self.log_dict(
            dictionary=self._loss_manager.loss_coef_log_dict,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        loss = self._loss_manager.get_loss()

        self.manual_backward(loss)

        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()

        return loss

    def validation_step(self, batch: Batch) -> torch.Tensor:

        batch = self.forward(batch=batch)

        self.log_dict(
            dictionary=self._loss_manager.loss_log_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        return self._loss_manager.get_loss()

    def configure_optimizers(self) -> dict:
        optimizers = []
        lr_schedulers_config = []

        for optimizer_cfg in self._optimizers_cfg:
            # config = {}

            # Generator is enough.
            params = (
                param
                for chain_link_name in optimizer_cfg.links
                for param in self._chain[chain_link_name].parameters(recurse=True)
            )
            optimizer = get_optimizer(
                optimizer_name=optimizer_cfg.optimizer_name,
                params=params,
                kwargs=vars(optimizer_cfg.kwargs),
            )
            # config["optimizer"] = optimizer
            optimizers.append(optimizer)

            if optimizer_cfg.lr_scheduler is not None:
                lr_scheduler = get_chained_scheduler(
                    optimizer=optimizer,
                    schedulers_cfg=optimizer_cfg.lr_scheduler.schedulers,
                )
                lr_scheduler_cfg = vars(
                    optimizer_cfg.lr_scheduler.out_cfg_kwargs
                )  # see configure_optimizers docs
                lr_scheduler_cfg["scheduler"] = lr_scheduler
                lr_schedulers_config.append(lr_scheduler_cfg)
                # config["lr_scheduler"] = lr_scheduler_cfg
            # configs.append(config)

        return optimizers, lr_schedulers_config

    def on_train_epoch_start(self) -> None:
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
        checkpoint["loss_manager"] = self._loss_manager.state_dict()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        loss_manager_state_dict = checkpoint["loss_manager"]
        self._loss_manager.load_state_dict(state_dict=loss_manager_state_dict)
