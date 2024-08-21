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
        file_path=Path("D4L-Hackaton") / "tests" / "utils" / "dummy_cfg-ChainAE.yaml"
    )
    chain_model = ChainAE(cfg)
    loss = chain_model.training_step(batch=batch_fixture)
    assert loss.requires_grad, "The loss should be a tensor with .requires_grad=True"
    loss.backward()
    chain_model.validation_step(batch=batch_fixture)


# def test_ChainModel_forward1(batch_fixture):

#     # Create a dummy configuration
#     cfg = _load_config_from_path(
#         file_path=Path("D4L-Hackaton") / "tests" / "utils" / "dummy_cfg-ChainModel.yaml"
#     )

#     # Create an instance of ChainModel
#     chain_model = ChainModel(cfg)
#     output = chain_model(batch_fixture)
#     assert isinstance(output["batch"]["data"], torch.Tensor)
#     assert output["batch"]["data"].shape == (3, 128)
#     losses_names = [loss["name"] for loss in output["losses"]]
#     assert "fuzz_clust" in losses_names
#     assert "comp_clust_reg" in losses_names


# def test_ChainModel_forward2(batch_fixture):

#     # Create a dummy configuration
#     cfg = _load_config_from_path(
#         file_path=Path("D4L-Hackaton")
#         / "tests"
#         / "utils"
#         / "dummy_cfg-ChainModel2.yaml"
#     )

#     # Create an instance of ChainModel
#     chain_model = ChainModel(cfg)
#     output = chain_model(batch_fixture)
#     assert isinstance(output["batch"]["data"], torch.Tensor)
#     assert output["batch"]["data"].shape == (3, 5, 128)
#     losses_names = [loss["name"] for loss in output["losses"]]
#     assert "fuzz_clust" in losses_names
#     assert "comp_clust_reg" in losses_names


# def test_ChainModel_run_command_pre_sample_embedding(batch_fixture):

#     # Create a dummy configuration
#     cfg = _load_config_from_path(
#         file_path=Path("D4L-Hackaton")
#         / "tests"
#         / "utils"
#         / "dummy_cfg-ChainModel3.yaml"
#     )

#     org_batch = batch_fixture.copy()
#     org_batch2 = batch_fixture.copy()

#     # Create an instance of ChainModel
#     chain_model = ChainModel(cfg)
#     output = chain_model(batch_fixture)
#     assert isinstance(output["batch"]["data"], torch.Tensor)
#     assert output["batch"]["data"].shape == (3, 5, 128)
#     losses_names = [loss["name"] for loss in output["losses"]]
#     assert "fuzz_clust" in losses_names
#     assert "comp_clust_reg" in losses_names

#     command_run_output = chain_model.run_processing_command(
#         batch=org_batch, command_name="pre_sample_embedding"
#     )
#     assert command_run_output["batch"]["data"].shape == (
#         org_batch2["data"].shape[0],
#         40,
#     )


# def test_ChainModel_run_command_sample_posterior(batch_fixture):

#     # Create a dummy configuration
#     cfg = _load_config_from_path(
#         file_path=Path("D4L-Hackaton")
#         / "tests"
#         / "utils"
#         / "dummy_cfg-ChainModel3.yaml"
#     )

#     org_batch = batch_fixture.copy()
#     # Create an instance of ChainModel
#     chain_model = ChainModel(cfg)

#     command_run_output = chain_model.run_processing_command(
#         batch=org_batch, command_name="sample_posterior"
#     )
#     assert command_run_output["batch"]["data"].shape == (
#         org_batch["data"].shape[0],
#         1,
#         20,
#     )
