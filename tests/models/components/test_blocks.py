import pytest
from src.models.components.blocks import (
    OrderedModuleSpecs,
    _get_module_from_spec,
    _wrap_block,
    Block,
    AddResidue,
    AddShortcut,
    BlockStack,
    ModuleSpec,
)
from src.utils.config import _load_config_from_path
from argparse import Namespace
import torch.nn as nn
from pathlib import Path
import torch


def test_get_module_from_spec_valid_spec():
    # Valid spec
    spec = ModuleSpec(
        **{"nnmodule_spec_path": "activation/relu", "kwargs": Namespace(**{})}
    )
    module = _get_module_from_spec(spec, 5)
    assert isinstance(module, nn.ReLU)


def test_get_module_from_spec_invalid_key1():
    # Invalid key1
    spec = ModuleSpec(**{"nnmodule_spec_path": "invalid_key/relu", "kwargs": {}})
    with pytest.raises(ValueError):
        _get_module_from_spec(spec, 5)


def test_get_module_from_spec_invalid_key2():
    # Invalid key2
    spec = ModuleSpec(**{"nnmodule_spec_path": "activation/invalid_key", "kwargs": {}})
    with pytest.raises(ValueError):
        _get_module_from_spec(spec, 6)


def test_block_forward_pass():
    # Define input tensor
    cfg = _load_config_from_path(
        file_path=Path("D4L-Hackaton")
        / "tests"
        / "utils"
        / "dummy_cfg2-block_stack.yaml"
    )
    ordered_module_specs = [
        ModuleSpec(**vars(spec)) for spec in cfg.ordered_module_specs
    ]
    block = Block(cfg.input_dim, cfg.output_dim, ordered_module_specs)

    # Generate random input tensor
    batch_size = 32
    input_tensor = torch.randn(batch_size, cfg.input_dim)

    # Perform forward pass
    output_tensor = block(input_tensor)

    # Check output tensor shape
    assert output_tensor.shape == (batch_size, cfg.output_dim)


def test_wrap_block_with_none_wrapper():
    # Wrap Block with None wrapper
    block_module = nn.Linear(10, 5)
    wrapped_block = _wrap_block(block_module, None)
    assert wrapped_block is block_module


def test_wrap_block_with_valid_wrapper():
    # Wrap Block with valid wrapper
    block_module = nn.Linear(10, 5)
    wrapped_block = _wrap_block(block_module, "residual")
    assert isinstance(wrapped_block, AddResidue)
    assert wrapped_block.module is block_module


def test_wrap_block_with_invalid_wrapper():
    # Wrap Block with invalid wrapper
    with pytest.raises(ValueError):
        _wrap_block(nn.Linear(10, 5), "invalid_wrapper")


if __name__ == "__main__":
    pytest.main()
