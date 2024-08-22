import pytest
from torch import Block
from src.utils.config import load_config_from_path, validate_config_structure
from src.models.components.blocks import BlockStack
from src.models.components.chain import ChainAE
from argparse import Namespace
from pathlib import Path


def test_load_config_from_path():
    # Test case 1: Valid file path
    # ikd why the path must be like that when module imports are done with src.
    file_path = Path("D4L-Hackaton") / "tests" / "utils" / "dummy_cfg1.yaml"

    assert file_path.exists()
    config = load_config_from_path(file_path)
    assert isinstance(config, Namespace)
    # Add more assertions for the expected behavior of the function with different inputs

    # Test case 2: Invalid file path
    with pytest.raises(FileNotFoundError):
        file_path = "/path/to/nonexistent.yaml"
        cfg = load_config_from_path(file_path)
    # Add more assertions for the expected behavior of the function with different inputs


def test_load_config_from_path_BlockStack():
    # Test case 1: Valid file path
    # ikd why the path must be like that when module imports are done with src.
    file_path = Path("D4L-Hackaton") / "tests" / "utils" / "dummy_cfg2-block_stack.yaml"

    assert file_path.exists()
    cfg = load_config_from_path(file_path)
    assert isinstance(cfg, Namespace)
    # Add more assertions for the expected behavior of the function with different inputs

    # Test case 2: Invalid file path
    with pytest.raises(FileNotFoundError):
        file_path = "/path/to/nonexistent.yaml"
        cfg = load_config_from_path(file_path)
    # Add more assertions for the expected behavior of the function with different inputs

    validate_config_structure(cfg=cfg, config_structure=BlockStack._config_structure)


def test_load_config_from_path_ChainModel():
    file_path = Path("D4L-Hackaton") / "tests" / "utils" / "dummy_cfg-ChainAE.yaml"

    assert file_path.exists()
    cfg = load_config_from_path(file_path)
    # Add more assertions for the expected behavior of the function with different inputs

    validate_config_structure(cfg=cfg, config_structure=ChainAE._config_structure)
