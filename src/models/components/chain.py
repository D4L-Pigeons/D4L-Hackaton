import torch
import torch.nn as nn
from src.utils.config import validate_config_structure, parse_choice_spec_path
from argparse import Namespace
from src.utils.common_types import ConfigStructure
from typing import List, Dict, TypeAlias, Type, TypedDict, Callable, Any
import inspect
from src.models.components.blocks import BlockStack, StandaloneTinyModule
from src.models.components.latent import (
    GaussianPosterior,
    GaussianMixturePriorNLL,
    LatentConstraint,
    FuzzyClustering,
    VectorConditionedLogitsGMPriorNLL,
)
from src.utils.common_types import (
    Batch,
    StructuredLoss,
    StructuredForwardOutput,
    format_structured_forward_output,
    format_structured_loss,
)
from src.models.components.misc import AggregateDataAdapter, BatchRearranger
from src.models.components.condition_embedding import ConditionEmbeddingTransformer
from src.models.components.loss import (
    get_reconstruction_loss,
    LossManager,
    map_loss_name,
)
import pytorch_lightning as pl
from src.models.components.optimizer import get_optimizer

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
    "batch_rearranger": BatchRearranger,
}

_CONDITION_EMBEDDING_MODULE: Dict[str, ChainLink] = {
    "condition_embedding_transformer": ConditionEmbeddingTransformer
}

_CHAIN_LINKS: Dict[str, Dict[str, ChainLink]] = {
    "blocks": _BLOCKS_MODULE,
    "latent": _LATENT_MODULE,
    "misc": _MISC_MODULE,
    "condition_embedding": _CONDITION_EMBEDDING_MODULE,
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
                f'The provided key {key} is wrong. Must be one of {", ".join(list(choice.keys()))}'
            )
        choice = next_choice
    return choice(cfg=chain_link_spec.cfg)


class ChainAE(pl.LightningModule):
    r"""
    ChainModel class represents a model that consists of a chain of chain links.
    Each chain link is responsible for processing a specific part of the input data.

    Args:
        cfg (Namespace): The configuration for the ChainModel.

    Attributes:
        _chain (nn.ModuleList): A list of chain links.

    Methods:
        forward(batch: Batch) -> StructuredForwardOutput:
            Performs forward pass through the ChainModel forwarding
            the batch through chain links and gathering the intermediate losses.

    """

    _config_structure: ConfigStructure = {
        "chain": [{"chain_link_spec_path": str, "name": str, "cfg": Namespace}],
        "optimizer": {"optimizer_name": str, "kwargs": Namespace},
        "loss": {
            "loss_manager": Namespace,
            "reconstruction_loss_spec": {
                "nnmodulename": str,
                "coef": float,
                "var_name": str,
                "kwargs": Namespace,
            },
        },
        "forward_link_order": [str],
        "processing_commands": [
            {
                "command_name": str,
                "processing_spec": {
                    "train": bool,
                    "steps": [
                        {"link_name": str, "method_name": str, "kwargs": Namespace}
                    ],
                },
            }
        ],
    }

    def __init__(self, cfg: Namespace) -> None:
        super(ChainAE, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._cfg: Namespace = cfg

        # Building chain based on the config.
        self._chain: nn.ModuleDict = nn.ModuleDict()
        self._setup_chain(chain_cfg=cfg.chain)

        # Setting the order of modules during forward.
        self._forward_link_order: List[str] = cfg.forward_link_order
        self._assert_forward_link_order()

        # Initializing LossMangager.
        self._loss_manager: LossManager = LossManager(cfg=cfg.loss.loss_manager)

        # Reconstruction loss is handled separately of the other losses in the ChainModel.
        self._reconstr_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
            get_reconstruction_loss(
                loss_name=cfg.loss.reconstruction_loss_spec.nnmodulename,
                kwargs=vars(cfg.loss.reconstruction_loss_spec.kwargs),
            )
        )
        self._reconstruction_loss_coef: float = cfg.loss.reconstruction_loss_spec.coef
        self._reconstruction_loss_aggregated: bool = not (
            vars(cfg.loss.reconstruction_loss_spec.kwargs).get("reduction", "none")
            == "none"
        )
        self._reconstruction_var_name: str = cfg.loss.reconstruction_loss_spec.var_name
        # Make the input tensor broadcastable if needed.
        self._input_tensor_dim_match_transform: callable = (
            (lambda x: x)
            if self._reconstruction_loss_aggregated
            else lambda x: x.unsqueeze(1)
        )

        # Preparing and setting the processing commands allowing to process the batch in a flexible manner.
        self._processing_commands: Dict[str, Dict[str, Dict | int]] = {}
        self._parse_processing_command_definitions(
            processing_commands=cfg.processing_commands
        )
        self._assert_commands()

    def _setup_chain(self, chain_cfg: List[Namespace]):
        for chain_spec in chain_cfg:
            assert (
                chain_spec.name not in self._chain
            ), f"The keyword {chain_spec.name} is already present in self._chain."
            self._chain[chain_spec.name] = _get_chain_link(chain_link_spec=chain_spec)

    def _assert_forward_link_order(self) -> None:
        for chain_link_name in self._forward_link_order:
            assert (
                chain_link_name in self._chain
            ), f"Chain link name {chain_link_name} is not present in self._chain = {list(self._chain.keys())}."

    def _parse_processing_command_definitions(
        self, processing_commands: List[Namespace]
    ) -> None:
        for command in processing_commands:
            self._processing_commands[command.command_name] = command.processing_spec

    def _assert_commands(self) -> None:
        for command_key, command_processing_spec in self._processing_commands.items():
            for cmd_proc_step in command_processing_spec.steps:
                assert (
                    cmd_proc_step.link_name in self._chain
                ), f"The {cmd_proc_step.link_name} does not match with any link_name from _chain {list(self._chain.keys())}."

                link = self._chain[cmd_proc_step.link_name]
                method = getattr(link, cmd_proc_step.method_name, None)

                assert (
                    method is not None
                ), f"Link '{cmd_proc_step.link_name}' has no attribute named {cmd_proc_step.method_name}."
                assert inspect.ismethod(method) or inspect.isfunction(
                    method
                ), f"Attribute '{cmd_proc_step.method_name}' of link '{cmd_proc_step.link_name}' is not a method."

    def run_processing_command(
        self,
        batch: Batch,
        command_name: str,
        dynamic_kwargs: Dict[str, Namespace] = {},
    ) -> Batch:
        assert (
            command_name in self._processing_commands
        ), f"The provivided command {command_name} is not within defined commands {self._processing_commands}."
        for link_name, kwargs in dynamic_kwargs.items():
            assert (
                link_name in self._chain
            ), f"The link of name '{link_name}' is not present in self._chain."
            assert isinstance(
                kwargs, Namespace
            ), "The provided kwargs are not an instance of Namespace."
        command_processing_spec = self._processing_commands[command_name]
        # Processing
        for processing_step in command_processing_spec.steps:
            chain_link = self._chain[processing_step.link_name]
            method = getattr(chain_link, processing_step.method_name)
            dynamic_kwargs_step = dynamic_kwargs.get(
                processing_step.link_name, Namespace()
            )
            chain_link_output = method(
                batch=batch,
                **vars(processing_step.kwargs),
                **vars(dynamic_kwargs_step),
            )
            assert (
                "batch" in chain_link_output
            ), f'Output of "{chain_link}" does not have "batch" key.'
            batch = chain_link_output["batch"]
        return batch

    def forward(self, batch: Batch) -> Batch:
        # Processing
        for link_name in self._forward_link_order:
            chain_link = self._chain[link_name]
            chain_link_output = chain_link(batch)
            assert (
                "batch" in chain_link_output
            ), f'Output of "{chain_link}" does not have "batch" key.'
            batch = chain_link_output["batch"]
            for structured_loss in chain_link_output["losses"]:
                self._loss_manager.process_structured_loss(structured_loss)
        return batch

    def _get_reconstruction_structured_loss(
        self, x_input: torch.Tensor, x_reconstr: torch.Tensor
    ) -> StructuredLoss:
        loss = self._reconstr_loss_fn(
            x_reconstr, self._input_tensor_dim_match_transform(x_input)
        )
        return format_structured_loss(
            loss=loss,
            name=map_loss_name("reconstruction"),
            coef=self._reconstruction_loss_coef,
            aggregated=self._reconstruction_loss_aggregated,
        )

    def training_step(self, batch: Batch) -> torch.Tensor:
        input_reconstr_var: torch.Tensor = batch[
            self._reconstruction_var_name
        ].clone()  # Is this clone necessary?

        batch = self.forward(batch=batch)

        output_reconstr_var: torch.Tensor = batch[self._reconstruction_var_name]
        reconstr_loss: StructuredLoss = self._get_reconstruction_structured_loss(
            x_input=input_reconstr_var, x_reconstr=output_reconstr_var
        )
        self._loss_manager.process_structured_loss(structured_loss=reconstr_loss)
        self.log_dict(
            dictionary=self._loss_manager.log_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return self._loss_manager.get_loss_aggregate()

    def validation_step(self, batch: Batch) -> torch.Tensor:
        input_reconstr_var: torch.Tensor = batch[self._reconstruction_var_name].clone()

        batch = self.forward(batch=batch)

        output_reconstr_var: torch.Tensor = batch[self._reconstruction_var_name]
        reconstr_loss: StructuredLoss = self._get_reconstruction_structured_loss(
            x_input=input_reconstr_var, x_reconstr=output_reconstr_var
        )
        self._loss_manager.process_structured_loss(structured_loss=reconstr_loss)
        self.log_dict(
            dictionary=self._loss_manager.log_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return self._loss_manager.get_loss_aggregate()

    def configure_optimizers(self):
        return get_optimizer(
            optimizer_name=self._cfg.optimizer.optimizer_name,
            params=self.parameters(),
            kwargs=vars(self._cfg.optimizer.kwargs),
        )
