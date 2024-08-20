from itertools import chain
from attr import validate
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
)
from src.models.components.misc import AggregateDataAdapter, BatchRearranger

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


class ChainModel(nn.Module):
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
    }

    def __init__(self, cfg: Namespace) -> None:
        super(ChainModel, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._chain: nn.ModuleList = nn.ModuleList()
        for chain_spec in cfg.chain:
            self._chain.append(_get_chain_link(chain_link_spec=chain_spec))

    def forward(self, batch: Batch) -> StructuredForwardOutput:
        losses: List[StructuredLoss] = []
        for chain_link in self._chain:
            chain_link_output = chain_link(batch)
            batch = chain_link_output["batch"]
            losses.extend(chain_link_output["losses"])
        return format_structured_forward_output(batch=batch, losses=losses)
