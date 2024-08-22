import torch
import torch.nn as nn
from src.utils.config import validate_config_structure, parse_choice_spec_path
from argparse import Namespace
from src.utils.common_types import ConfigStructure
from typing import List, Dict, TypeAlias, Type, TypedDict
from src.models.components.blocks import BlockStack
from src.models.components.latent import (
    GaussianPosterior,
    GaussianMixturePriorNLL,
    LatentConstraint,
    FuzzyClustering,
)
from src.utils.common_types import (
    Batch,
    StructuredLoss,
    StructuredForwardOutput,
    format_structured_forward_output,
    format_structured_loss,
)
from src.models.components.misc import AggregateDataAdapter, BatchRearranger
from typing import Callable
from src.models.components.loss import (
    get_reconstruction_loss,
    LossManager,
    map_loss_name,
)
import pytorch_lightning as pl
from src.models.components.optimizer import get_optimizer

ChainLink: TypeAlias = Type[nn.Module]

_BLOCKS_MODULE: Dict[str, ChainLink] = {"block_stack": BlockStack}

_LATENT_MODULE: Dict[str, ChainLink] = {
    "posterior_gaussian": GaussianPosterior,
    "prior_gaussian_mixture_nll": GaussianMixturePriorNLL,
    "constraint_latent": LatentConstraint,
    "clustering_fuzzy": FuzzyClustering,
}
_MISC_MODULE: Dict[str, ChainLink] = {
    "adapter": AggregateDataAdapter,
    "batch_rearranger": BatchRearranger,
}

_CHAIN_LINKS: Dict[str, Dict[str, ChainLink]] = {
    "blocks": _BLOCKS_MODULE,
    "latent": _LATENT_MODULE,
    "misc": _MISC_MODULE,
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
                f'The provided key {key} is wrong. Must be one of {" ,".join(list(choice.keys()))}'
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
        "chain": [{"chain_link_spec_path": str, "cfg": Namespace}],
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
        "processing_commands": [
            {"command_name": str, "final_link_idx": int, "kwargs": Namespace}
        ],
    }

    def __init__(self, cfg: Namespace) -> None:
        super(ChainAE, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        # Preparing and setting the processing commands allowing to process the batch up to the chain link specified by index.
        self._processing_commands: Dict[str, Dict[str, Dict | int]] = {}
        self._parse_processing_command_definitions(
            processing_commands=cfg.processing_commands
        )

        # Building chain based on the config.
        self._chain: nn.ModuleList = nn.ModuleList()
        self._build_chain(chain_cfg=cfg.chain)

        # Checking the command indices correctness.
        self._assert_commands()

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

    def _parse_processing_command_definitions(
        self, processing_commands: List[Namespace]
    ) -> None:
        for command in processing_commands:
            self._processing_commands[command.command_name] = {
                "final_link_idx": command.final_link_idx,
                "kwargs": vars(command.kwargs),
            }

    def _build_chain(self, chain_cfg: List[Namespace]):
        for chain_spec in chain_cfg:
            self._chain.append(_get_chain_link(chain_link_spec=chain_spec))

    def _assert_commands(self) -> None:
        for command_key, command_val in self._processing_commands.items():
            assert command_val["final_link_idx"] < len(
                self._chain
            ), f"The value {command_val['final_link_idx']} for command {command_key} if greater than the number of chain_links={len(self._chain)}."

    def run_processing_command(self, batch: Batch, command_name: str) -> Batch:
        assert (
            command_name in self._processing_commands
        ), f"The provivided command {command_name} is not within defined commands {self._processing_commands}."
        command = self._processing_commands[command_name]
        final_link_idx: int = command["final_link_idx"]

        # Processing up to the final link specified in command definition without passing on loss information.
        for chain_link in self._chain[:final_link_idx]:
            chain_link_output = chain_link(batch)
            batch = chain_link_output["batch"]

        # Special treatment for the last link, which may take additional kwargs to the forward method.
        chain_link_output = self._chain[final_link_idx](
            batch=batch, **command["kwargs"]
        )
        batch = chain_link_output["batch"]

        return batch

    def _get_reconstruction_structured_loss(
        self, x_input: torch.Tensor, x_reconstr: torch.Tensor
    ) -> StructuredLoss:
        loss = self._reconstr_loss_fn(
            self._input_tensor_dim_match_transform(x_input), x_reconstr
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

        # Processing
        for chain_link in self._chain:
            chain_link_output = chain_link(batch)
            batch = chain_link_output["batch"]
            for structured_loss in chain_link_output["losses"]:
                self._loss_manager.process_structured_loss(structured_loss)

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
        for chain_link in self._chain:
            chain_link_output = chain_link(batch)
            batch = chain_link_output["batch"]
            for structured_loss in chain_link_output["losses"]:
                self._loss_manager.process_structured_loss(structured_loss)

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
            optimizer_name=self.cfg.optimizer.optimizer_name,
            kwargs=vars(self.cfg.optimizer.kwargs),
        )
