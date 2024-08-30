from argparse import Namespace
import pytest
import torch
from src.models.components.chain import ChainAE
from src.utils.common_types import Batch, StructuredForwardOutput
from src.utils.config import load_config_from_path
from pathlib import Path


@pytest.fixture
def batch_fixture():
    batch_size, dim = 3, 128
    tensor = torch.arange(batch_size * dim, dtype=torch.float32).reshape(
        batch_size, dim
    )
    return {
        "data": tensor,
        "component_indicator": torch.tensor([-1, 0, 1]),
    }


def test_ChainAE(batch_fixture):
    cfg = load_config_from_path(
        file_path=Path("D4L-Hackaton")
        / "tests"
        / "models"
        / "components"
        / "dummy_cfgs"
        / "dummy_cfg-ChainAE.yaml"
    )
    chain_model = ChainAE(cfg)
    loss = chain_model.training_step(batch=batch_fixture)
    assert loss.requires_grad, "The loss should be a tensor with .requires_grad=True"
    loss.backward()
    chain_model.validation_step(batch=batch_fixture)


def test_ChainAE_command(batch_fixture):
    cfg = load_config_from_path(
        file_path=Path("D4L-Hackaton")
        / "tests"
        / "models"
        / "components"
        / "dummy_cfgs"
        / "dummy_cfg-ChainAE-command.yaml"
    )
    chain_model = ChainAE(cfg)
    output = chain_model.run_processing_command(
        batch=batch_fixture, command_name="full_forward"
    )
    assert output["data"].shape == (3, 128)
    output = chain_model.run_processing_command(
        batch=batch_fixture, command_name="partial_forward"
    )
    assert output["data"].shape == (3, 40)
