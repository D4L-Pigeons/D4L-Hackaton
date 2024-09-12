r"""
This file contains the implementation of dense blocks i.e. MLP components used in the Chain class as so called chain links in the Chain nomenclature.

- `StandaloneTinyNNModule`: Represents a standalone tiny module used for applying a tiny module outside of `BlockStack` e.g. an activation applied to the output of the model.
- `DenseBlockStack`: Represents a stack of dense blocks, used for building MLPs in a configurable way.

"""

import torch
import torch.nn as nn
from argparse import Namespace
from typing import List, Dict, Type, TypeAlias, Optional, NamedTuple, Any

from utils.config import (
    validate_config_structure,
    parse_choice_spec_path,
)
from utils.common_types import (
    Batch,
    ConfigStructure,
    StructuredForwardOutput,
    format_structured_forward_output,
)


class TinyNNModuleSpec(Namespace):
    r"""
    TinyModuleSpec class represents a specification for a "tiny module"
    e.g. activation function, batch norm layer, dropout etc. used as a sumbodues of a Block module.

    Attributes:
        nnmodule_spec_path (str): The path to the nn.Module in a _TINY_MODULES dict.
        kwargs (Namespace): Additional keyword arguments passsed to the nn.Module's constructor.
    """

    nnmodule_spec_path: str
    kwargs: Namespace


OrderedTinyNNModuleSpecs: TypeAlias = List[TinyNNModuleSpec]


class TinyNNModuleWithSpecialTreatment(NamedTuple):
    r"""
    A named tuple wrapper over nn.Module used for inputting out_features of the nn.Linear with specified keyword.

    Attributes:
        nnmodule (nn.Module): Module which takes out_features=output_dim as a kwarg.
        out_features_keyword (str): The nn.Module keyword which takes value output_dim .
    """

    nnmodule: nn.Module
    out_features_keyword: str


DictofTinyNNModules: TypeAlias = Dict[
    str, Type[nn.Module] | TinyNNModuleWithSpecialTreatment
]


_ACTIVATION: DictofTinyNNModules = {
    "relu": nn.ReLU,
    "prelu": nn.PReLU,
    "rrelu": nn.RReLU,
    "celu": nn.CELU,
    "selu": nn.SELU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}

_NORM: DictofTinyNNModules = {
    "batch": TinyNNModuleWithSpecialTreatment(
        nnmodule=nn.BatchNorm1d, out_features_keyword="num_features"
    ),
    "layer": nn.LayerNorm,
}

_DROPOUT: DictofTinyNNModules = {"ordinary": nn.Dropout}

_TINY_MODULES: Dict[str, DictofTinyNNModules] = {
    "activation": _ACTIVATION,
    "norm": _NORM,
    "dropout": _DROPOUT,
}


def _get_tiny_module_from_spec(
    spec: TinyNNModuleSpec, output_dim: Optional[int]
) -> nn.Module:
    r"""
    Retrieves a nn.Module for the given specification.
    Args:
        spec (TinyNNModuleSpec): The specification of the tiny module.
        output_dim (Optional[int]): The out_features arguement of the nn.Linear.
    Returns:
        nn.Module: The instantiated tiny module.
    Raises:
        ValueError: If the provided key is wrong or if the output dimension is not provided.
        AssertionError: If the spec argument is not an instance of TinyModuleSpec.
    """

    assert isinstance(
        spec, TinyNNModuleSpec
    ), f"Argument spec is of type {type(spec)} not an instance of ModuleSpec."

    choice_path: List[str] = parse_choice_spec_path(spec_path=spec.nnmodule_spec_path)
    choice = _TINY_MODULES

    for key in choice_path:
        next_choice = choice.get(key, None)

        if next_choice is None:
            raise ValueError(
                f'The provided key {key} is wrong. Must be one of {" ,".join(list(choice.keys()))}'
            )

        choice = next_choice

    kwargs = vars(spec.kwargs) if spec.kwargs is not None else {}

    if isinstance(choice, TinyNNModuleWithSpecialTreatment):
        if output_dim is None:
            raise ValueError("Argument output_dim needs to be provided.")
        kwargs[choice.output_dim_keyword] = output_dim
        choice = choice.nnmodule

    return choice(**kwargs)


class StandaloneTinyNNModule(nn.Module):
    r"""
    StandaloneTinyNNModule(nn.Module)

    This class represents a standalone tiny module i.e. an nn.Module which may be used as a chain link in Chain class
    similarly to BlockStack. Used to apply a tiny module outside BlockStack e.g. for for output activation.

    Parameters:
    - cfg (Namespace): The configuration for the nn.Module.

    Methods:
    - forward(batch: Batch, data_name: str) -> StructuredForwardOutput:
        Performs the forward pass of the module.

    Static Methods:
    - _parse_hparams_to_dict(cfg: Namespace) -> Dict[str, Any]:
        Parses the hyperparameters to a dictionary for logging purposes.

    Attributes:
    - _config_structure (ConfigStructure): The configuration structure for the module.
    - _tiny_module (nn.Module): The tiny module used in the forward pass.

    """

    _config_structure: ConfigStructure = {
        "nnmodule_spec_path": str,
        "kwargs": Namespace,
        "output_dim": (None, int),
    }

    def __init__(self, cfg: Namespace) -> None:
        super(StandaloneTinyNNModule, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        spec: TinyNNModuleSpec = TinyNNModuleSpec(
            nnmodule_spec_path=cfg.nnmodule_spec_path, kwargs=cfg.kwargs
        )
        self._tiny_module: nn.Module = _get_tiny_module_from_spec(
            spec=spec, output_dim=cfg.output_dim
        )

    @staticmethod
    def _parse_hparams_to_dict(cfg: Namespace) -> Dict[str, Any]:
        return {
            "nnmodule_spec_path": cfg.nnmodule_spec_path,
            "kwargs": vars(cfg.kwargs),
            "output_dim": cfg.output_dim,
        }

    def forward(self, batch: Batch, data_name: str) -> StructuredForwardOutput:
        batch[data_name] = self._tiny_module(batch[data_name])

        return format_structured_forward_output(batch=batch)


class DenseBlock(nn.Module):
    r"""
    A module that consists of a linear layer followed by a sequence of additional tiny modules.
    Args:
        input_dim (int): The input dimension of the linear layer.
        output_dim (int): The output dimension of the linear layer.
        ordered_module_specs (OrderedTinyNNModuleSpecs): The ordered specifications of additional tiny modules.
    Attributes:
        input_dim (int): The input dimension of the linear layer.
        output_dim (int): The output dimension of the linear layer.
        layer (nn.ModuleList): The list of modules consisting of a linear layer and additional tiny modules.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Performs forward pass through the dense block module.
    Static Methods:
        _check_if_bias(ordered_module_specs: OrderedTinyNNModuleSpecs) -> bool:
            Checks if biases should be learned based on the ordered module specifications.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        ordered_module_specs: OrderedTinyNNModuleSpecs,
    ) -> None:
        super(DenseBlock, self).__init__()

        self.input_dim: int = input_dim
        self.output_dim: int = output_dim

        # Setting up nn.Linear with a list of additional tiny modules specified in the ordered_module_specs.
        self.layer = nn.ModuleList(
            [
                nn.Linear(
                    in_features=input_dim,
                    out_features=output_dim,
                    bias=self._check_if_bias(
                        ordered_module_specs=ordered_module_specs
                    ),  # If the batch norm is used we are not learning biases.
                ),
            ]
        )

        # Initializing nn.Linear weights
        torch.nn.init.xavier_normal_(self.layer[0].weight)

        # Adding specified tiny modules in the specified order after the nn.Linear.
        for spec in ordered_module_specs:
            self.layer.append(
                _get_tiny_module_from_spec(spec=spec, output_dim=output_dim)
            )

    @staticmethod
    def _check_if_bias(ordered_module_specs: OrderedTinyNNModuleSpecs) -> bool:
        return not any(
            [
                module_spec.nnmodule_spec_path == "norm/batch"
                for module_spec in ordered_module_specs
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_component in self.layer:
            x = layer_component(x)
        return x


class AddResidue(nn.Module):
    r"""
    AddResidue module that adds the input tensor to the output of a given module.

    Args:
        module (nn.Module): The module to add the input tensor to.

    Returns:
        torch.Tensor: The output tensor after adding the input tensor to the module's output.
    """

    def __init__(self, module: nn.Module) -> None:
        super(AddResidue, self).__init__()

        self.module: nn.Module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)


class AddShortcut(nn.Module):
    r"""
    A module that adds a shortcut connection to the input tensor.
    Args:
        module (nn.Module): The module to which the shortcut connection is added.
    Attributes:
        module (nn.Module): The module to which the shortcut connection is added.
        shortcut (nn.Linear): The linear layer representing the shortcut connection.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Performs the forward pass of the module.
    """

    def __init__(self, module: nn.Module) -> None:
        super(AddShortcut, self).__init__()

        self.module: nn.Module = module
        self.shortcut: nn.Linear = nn.Linear(
            input_size=module.input_dim, output_size=module.output_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.block(x)


_STACKED_DENSE_BLOCKS_WRAPPERS: Dict[str, Type[nn.Module]] = {
    "residual": AddResidue,
    "shortcut": AddShortcut,
}


def _wrap_stacked_dense_blocks(
    block_module: nn.Module, block_wrapper_name: Optional[str]
) -> nn.Module:
    r"""
    Wraps the given `block_module` with a block wrapper based on the provided `block_wrapper_name`.

    Args:
        block_module (nn.Module): The dense block module to be wrapped.
        block_wrapper_name (Optional[str]): The name of the block wrapper to be used. If None, the original `block_module` is returned.

    Returns:
        nn.Module: The wrapped dense block module.

    Raises:
        ValueError: If the provided `block_wrapper_name` is not one of the available block wrappers.
    """

    if block_wrapper_name == None:
        return block_module

    block_wrapper = _STACKED_DENSE_BLOCKS_WRAPPERS.get(block_wrapper_name, None)

    if block_wrapper is None:
        raise ValueError(
            f'The provided block_wrapper_name {block_wrapper_name} is wrong. Must be one of {" ,".join(list(_STACKED_DENSE_BLOCKS_WRAPPERS.keys()))}'
        )

    return block_wrapper(block_module)


class DenseBlockStack(nn.Module):
    r"""
    A module that represents a stack of DenseBlocks i.e. a MLP with a list
    of specified nn.Modules after each but last nn.Linear.

    Args:
        cfg (Namespace): The configuration for the dense block stack.

    Attributes:
        input_dim (int): The input dimension of the dense block stack.
        output_dim (int): The output dimension of the dense block stack.
        blocks (nn.Sequential): The sequential dense blocks in the stack.

    Methods:
        forward(batch: Batch, data_name: str) -> StructuredForwardOutput:
            Performs a forward pass through the dense block stack.

    Static Methods:
        _parse_hparams_to_dict(cfg: Namespace) -> Dict[str, Any]:
            Parses the hyperparameters to a dictionary.

    """

    _config_structure: ConfigStructure = {
        "input_dim": int,
        "output_dim": int,
        "hidden_dims": list,
        "ordered_module_specs": list,
        "block_wrapper_name": str | None,
    }

    def __init__(self, cfg: Namespace) -> None:
        super(DenseBlockStack, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self.input_dim: int = cfg.input_dim
        self.output_dim: int = cfg.output_dim

        blocks: List[nn.Module] = []
        dims: List[int] = [cfg.input_dim] + cfg.hidden_dims

        ordered_module_specs: OrderedTinyNNModuleSpecs = [
            TinyNNModuleSpec(**vars(spec))
            for spec in cfg.ordered_module_specs  # Conversion from List[Namespace] to List[TinyNNModuleSpec] done just for consistency.
        ]

        # Creating blocks if len(dims) > 1 otherwise just a single nn.Linear is added.
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            blocks.append(
                DenseBlock(
                    input_dim=in_dim,
                    output_dim=out_dim,
                    ordered_module_specs=ordered_module_specs,
                ),
            )

        blocks.append(nn.Linear(in_features=dims[-1], out_features=cfg.output_dim))

        # Wrapping stacked dense blocks if the wrapper is specified.
        blocks: Type[nn.Module] = _wrap_stacked_dense_blocks(
            block_module=nn.Sequential(*blocks),
            block_wrapper_name=cfg.block_wrapper_name,
        )

        # Setting nn.Module attribute used in forward.
        self.blocks: nn.Sequential = nn.Sequential(*blocks)

    @staticmethod
    def _parse_hparams_to_dict(cfg: Namespace) -> Dict[str, Any]:
        return {
            "input_dim": cfg.input_dim,
            "output_dim": cfg.output_dim,
            "hidden_dims": str(cfg.hidden_dims),
            "ordered_module_specs": str(cfg.ordered_module_specs),
            "block_wrapper_name": cfg.block_wrapper_name,
        }

    def forward(self, batch: Batch, data_name: str) -> StructuredForwardOutput:
        batch[data_name] = self.blocks(batch[data_name])

        return format_structured_forward_output(batch=batch, losses=[])
