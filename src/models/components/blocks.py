from argparse import Namespace
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any, Type


_ACTIVATION: Dict[str, nn.Module] = {
    "relu": nn.ReLU,
    "prelu": nn.PReLU,
    "rrelu": nn.RReLU,
    "celu": nn.CELU,
    "selu": nn.SELU,
}

_NORM: Dict[str, Type[nn.Module]] = {
    "batch": nn.BatchNorm1d,
}

_DROPOUT: Dict[str, Type[nn.Module]] = {"ordinary": nn.Dropout}

_SPECS: Dict[str, Dict[str, Type[nn.Module]]] = {
    "activation": _ACTIVATION,
    "norm": _NORM,
    "dropout": _DROPOUT,
}


def _get_module_from_spec(spec: Dict[str, str | Dict[str, Any]]) -> nn.Module:
    return _SPECS[spec["key"]][spec["key"]["meta_kwargs"]["type"]](
        **spec["key"]["kwargs"]
    )


class Block(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, cfg: Namespace) -> None:
        r"""
        Initialize the Block object.

        Args:
            input_dim (int): The input dimension.
            output_dim (int): The output dimension.
            cfg (Namespace): The configuration object.
                - ordered_spec (List[Dict[str, str | Dict[Any]]])
                e.g.
                    [
                    {"key": "norm", "meta_kwargs": {"type": "batch"}, "kwargs": {}},
                    {"key": "activation", "meta_kwargs": {"type": "relu"}, "kwargs": {}},
                    {"key": "dropout", "meta_kwargs": {"type": "ordinary"}, "kwargs": {}}
                    ]
                    }

        Returns:
            None
        """
        super(Block, self).__init__()
        self.intput_dim: int = input_dim
        self.output_dim: int = output_dim
        self.layer = nn.ModuleList(
            nn.Linear(
                input_dim,
                output_dim,
                bias=not "norm"
                in any(
                    [
                        item["key"] == "norm" and item["key"]["type"] == "batch"
                        for item in cfg.ordered_spec
                    ]
                ),  # If the batch norm is used we are not learning biases.
            )
        )
        # Adding the specified modules in the order.
        for spec in cfg.ordered_spec:
            self.layer.append(_get_module_from_spec(spec=spec))

    def forward(self, x: Tensor) -> Tensor:
        for layer_component in self.layer:
            x = layer_component(x)
        return x


class AddResidue(nn.Module):

    def __init__(self, module: nn.Module) -> None:
        super(AddResidue, self).__init__()
        self.module = module

    def forward(self, x: Tensor) -> Tensor:
        return x + self.module(x)


class AddShortcut(nn.Module):

    def __init__(self, module: nn.Module) -> None:
        super(AddShortcut, self).__init__()
        self.module = module
        self.shortcut = nn.Linear(
            input_size=module.input_dim, output_size=module.output_dim
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.shortcut(x) + self.block(x)


class BlockStack(nn.Module):
    def __init__(self, block_constructor: Type[nn.Module], cfg: Namespace) -> None:
        r"""
        Initializes a BlockStack object.

        Args:
            block_constructor (Type[nn.Module]): The constructor for the block module.
            cfg (Namespace): The configuration namespace.
                - input_dim (int): The dimensionality of the input.
                - hidden_dims (List[int]): The dimensions of the hidden layers.
                - output_dim (int): The dimensionality of the output.
                - block (Namespace): Configuration of the blocks.

        Returns:
            None
        """
        super(BlockStack, self).__init__()
        self.input_dim: int = cfg.input_dim
        self.output_dim: int = cfg.output_dim
        self.blocks = nn.ModuleList()
        dims = [cfg.input_dim] + cfg.hidden_dims

        # Creating blocks if len(dims) > 1
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.block.append(
                block_constructor(input_dim=in_dim, output_dim=out_dim, cfg=cfg.block)
            )

        self.blocks.append(nn.Linear(in_features=dims[-1], out_features=cfg.output_dim))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.blocks:
            x = layer(x)
        return self.out(x)
