from argparse import Namespace
import pytest
import torch
from src.models.components.chain_model import ChainModel
from src.utils.common_types import Batch, StructuredForwardOutput
from src.utils.config import _load_config_from_path
from pathlib import Path


def test_ChainModel():

    # Create a dummy configuration
    cfg = _load_config_from_path(
        file_path=Path("D4L-Hackaton") / "tests" / "utils" / "dummy_cfg-ChainModel.yaml"
    )

    # Create an instance of ChainModel
    chain_model = ChainModel(cfg)


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


def test_ChainModel_forward1(batch_fixture):

    # Create a dummy configuration
    cfg = _load_config_from_path(
        file_path=Path("D4L-Hackaton") / "tests" / "utils" / "dummy_cfg-ChainModel.yaml"
    )

    # Create an instance of ChainModel
    chain_model = ChainModel(cfg)
    output = chain_model(batch_fixture)
    assert isinstance(output["batch"]["data"], torch.Tensor)
    assert output["batch"]["data"].shape == (3, 128)
    losses_names = [loss["name"] for loss in output["losses"]]
    assert "fuzz_clust" in losses_names
    assert "comp_clust_reg" in losses_names


def test_ChainModel_forward2(batch_fixture):

    # Create a dummy configuration
    cfg = _load_config_from_path(
        file_path=Path("D4L-Hackaton")
        / "tests"
        / "utils"
        / "dummy_cfg-ChainModel2.yaml"
    )

    # Create an instance of ChainModel
    chain_model = ChainModel(cfg)
    output = chain_model(batch_fixture)
    assert isinstance(output["batch"]["data"], torch.Tensor)
    assert output["batch"]["data"].shape == (3, 5, 128)
    losses_names = [loss["name"] for loss in output["losses"]]
    assert "fuzz_clust" in losses_names
    assert "comp_clust_reg" in losses_names