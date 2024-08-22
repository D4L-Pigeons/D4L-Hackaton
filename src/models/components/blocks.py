from argparse import Namespace
from typing import (
    Any,
    List,
    Dict,
    Type,
    TypeAlias,
    TypedDict,
    Optional,
    Union,
    NamedTuple,
)
from numpy import block, isin
import torch.nn as nn
from torch import Tensor
from src.utils.common_types import (
    Batch,
    ConfigStructure,
    StructuredForwardOutput,
    format_structured_forward_output,
)
from src.utils.config import (
    validate_config_structure,
    parse_choice_spec_path,
)
from src.utils.common_types import format_structured_forward_output


class ModuleSpec(Namespace):
    nnmodule_path: str
    kwargs: Namespace


OrderedModuleSpecs: TypeAlias = List[ModuleSpec]


class ModuleWithSpecialTreatment(NamedTuple):
    nnmodule: nn.Module
    output_dim_keyword: str


DictofModules: TypeAlias = Dict[str, Type[nn.Module] | ModuleWithSpecialTreatment]


_ACTIVATION: DictofModules = {
    "relu": nn.ReLU,
    "prelu": nn.PReLU,
    "rrelu": nn.RReLU,
    "celu": nn.CELU,
    "selu": nn.SELU,
}

_NORM: DictofModules = {
    "batch": ModuleWithSpecialTreatment(
        nnmodule=nn.BatchNorm1d, output_dim_keyword="num_features"
    ),
}

_DROPOUT: DictofModules = {"ordinary": nn.Dropout}

_MODULE_SPECS: Dict[str, DictofModules] = {
    "activation": _ACTIVATION,
    "norm": _NORM,
    "dropout": _DROPOUT,
}


def _get_module_from_spec(spec: ModuleSpec, output_dim: int) -> nn.Module:
    r"""
    Get a module from the provided specification.

    Args:
        spec (ModuleSpec): The specification of the module.

    Returns:
        nn.Module: The module corresponding to the specification.

    Raises:
        ValueError: If the provided spec["key1"] is not valid.
        ValueError: If the provided spec["key2"] is not valid.
    """
    assert isinstance(
        spec, ModuleSpec
    ), f"Argument spec is of type {type(spec)} not an instance of ModuleSpec."

    choice_path: List[str] = parse_choice_spec_path(spec_path=spec.nnmodule_spec_path)
    choice = _MODULE_SPECS
    for key in choice_path:
        next_choice = choice.get(key, None)
        if next_choice is None:
            raise ValueError(
                f'The provided key {key} is wrong. Must be one of {" ,".join(list(choice.keys()))}'
            )
        choice = next_choice
    kwargs = vars(spec.kwargs) if spec.kwargs is not None else {}
    if isinstance(choice, ModuleWithSpecialTreatment):
        kwargs[choice.output_dim_keyword] = output_dim
        choice = choice.nnmodule
    return choice(**kwargs)


class Block(nn.Module):
    r"""
    A block module that consists of multiple layers.
    Args:
        input_dim (int): The input dimension of the block.
        output_dim (int): The output dimension of the block.
        ordered_module_specs (OrderedModuleSpecs): The ordered module specifications.
    Attributes:
        intput_dim (int): The input dimension of the block.
        output_dim (int): The output dimension of the block.
        layer (nn.ModuleList): The list of layers in the block.
    Methods:
        forward(x: Tensor) -> Tensor: Performs forward pass through the block.
    """

    def __init__(
        self, input_dim: int, output_dim: int, ordered_module_specs: OrderedModuleSpecs
    ) -> None:
        super(Block, self).__init__()

        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.layer = nn.ModuleList(
            [
                nn.Linear(
                    input_dim,
                    output_dim,
                    bias=self._check_if_bias(ordered_module_specs=ordered_module_specs),
                ),
            ]
        )
        # Adding the specified modules in the order.
        for spec in ordered_module_specs:
            self.layer.append(_get_module_from_spec(spec=spec, output_dim=output_dim))

    @staticmethod
    def _check_if_bias(ordered_module_specs: OrderedModuleSpecs) -> bool:
        return not any(
            [
                module_spec.nnmodule_spec_path == "norm/batch"
                for module_spec in ordered_module_specs
            ]
        )  # If the batch norm is used we are not learning biases.

    def forward(self, x: Tensor) -> Tensor:
        for layer_component in self.layer:
            x = layer_component(x)
        return x


class AddResidue(nn.Module):
    r"""
    AddResidue module that adds the input tensor to the output of the given module.

    Args:
        module (nn.Module): The module to add the input tensor to.

    Returns:
        Tensor: The output tensor after adding the input tensor to the module's output.
    """

    def __init__(self, module: nn.Module) -> None:
        super(AddResidue, self).__init__()
        self.module = module

    def forward(self, x: Tensor) -> Tensor:
        return x + self.module(x)


class AddShortcut(nn.Module):
    r"""
    Module that adds a shortcut connection to the input tensor.

    Args:
        module (nn.Module): The module to which the shortcut connection is added.

    Attributes:
        module (nn.Module): The module to which the shortcut connection is added.
        shortcut (nn.Linear): The linear layer representing the shortcut connection.

    Methods:
        forward(x: Tensor) -> Tensor:
            Performs the forward pass of the module.

    """

    def __init__(self, module: nn.Module) -> None:
        super(AddShortcut, self).__init__()
        self.module = module
        self.shortcut = nn.Linear(
            input_size=module.input_dim, output_size=module.output_dim
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.shortcut(x) + self.block(x)


_BLOCK_WRAPPERS: Dict[str, Type[nn.Module]] = {
    "residual": AddResidue,
    "shortcut": AddShortcut,
}


def _wrap_block(
    block_module: nn.Module, block_wrapper_name: Optional[str]
) -> nn.Module:
    r"""
    Wraps a block module with a specified block wrapper.

    Args:
        block_module (Type[nn.Module]): The block module to be wrapped.
        block_wrapper_name (Optional[str]): The name of the block wrapper. If None, the original block module is returned.

    Returns:
        Type[nn.Module]: The wrapped block module.

    Raises:
        ValueError: If the provided block_wrapper_name is not found in the available block wrappers.

    """
    if block_wrapper_name == None:
        return block_module
    block_wrapper = _BLOCK_WRAPPERS.get(block_wrapper_name, None)
    if block_wrapper is not None:
        return block_wrapper(block_module)
    raise ValueError(
        f'The provided block_wrapper_name {block_wrapper_name} is wrong. Must be one of {" ,".join(list(_BLOCK_WRAPPERS.keys()))}'
    )


class BlockStack(nn.Module):
    r"""
    A stack of blocks used in a neural network model.

    Args:
        cfg (Namespace): The configuration namespace containing the input and output dimensions.

    Attributes:
        input_dim (int): The input dimension of the block stack.
        output_dim (int): The output dimension of the block stack.
        blocks (ModuleList): A list of blocks in the stack.

    """

    _config_structure: ConfigStructure = {
        "input_dim": int,
        "output_dim": int,
        "hidden_dims": list,
        "ordered_module_specs": list,
        "data_name": str,
        "block_wrapper_name": str | None,
    }

    def __init__(self, cfg: Namespace) -> None:
        super(BlockStack, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self.input_dim: int = cfg.input_dim
        self.output_dim: int = cfg.output_dim
        self._data_name: str = cfg.data_name

        blocks: List[nn.Module] = []
        dims: List[int] = [cfg.input_dim] + cfg.hidden_dims

        ordered_module_specs: OrderedModuleSpecs = [
            ModuleSpec(**vars(spec))
            for spec in cfg.ordered_module_specs  # Conversion from List[Namespace] done just for consistency.
        ]
        # Creating blocks if len(dims) > 1
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            blocks.append(
                Block(
                    input_dim=in_dim,
                    output_dim=out_dim,
                    ordered_module_specs=ordered_module_specs,
                ),
            )

        blocks.append(nn.Linear(in_features=dims[-1], out_features=cfg.output_dim))
        blocks: Type[nn.Module] = _wrap_block(
            block_module=nn.Sequential(*blocks),
            block_wrapper_name=cfg.block_wrapper_name,
        )
        self.blocks: nn.Sequential = nn.Sequential(*blocks)

    def forward(self, batch: Batch) -> StructuredForwardOutput:
        batch[self._data_name] = self.blocks(batch[self._data_name])
        return format_structured_forward_output(batch=batch, losses=[])
