import pytest
import torch
from src.models.components.misc import BatchRearranger, AggregateDataAdapter
from src.utils.config import load_config_from_path
from pathlib import Path


@pytest.fixture
def sample_batch():
    # Define a sample batch for testing
    batch_size, n_samples, dim = 4, 6, 7
    batch = {
        "data1": torch.arange(batch_size * n_samples * dim).reshape(
            batch_size, n_samples, dim
        ),
        "data2": torch.arange(batch_size * n_samples * dim).reshape(
            batch_size, n_samples, dim
        ),
        "data3": torch.arange(batch_size * n_samples * dim).reshape(
            batch_size, n_samples, dim
        ),
    }
    return batch


def test_batch_rearranger(sample_batch):
    # Define the rearrange patterns
    cfg = load_config_from_path(
        file_path=Path("D4L-Hackaton")
        / "tests"
        / "utils"
        / "dummy_cfg-BatchRearranger.yaml"
    )

    # Create an instance of the BatchRearranger
    rearranger = BatchRearranger(cfg=cfg)

    org_sample_batch = sample_batch.copy()

    # Perform forward pass
    output_batch = rearranger(batch=sample_batch)["batch"]

    assert "data1" in output_batch

    # Check if the data is rearranged correctly
    assert torch.all(
        torch.eq(
            output_batch["data1"],
            org_sample_batch["data1"].permute([0, 1, 2]),
        )
    )
    assert torch.all(
        torch.eq(
            output_batch["data2"],
            org_sample_batch["data2"].permute([1, 0, 2]),
        )
    )
    assert torch.all(
        torch.eq(
            output_batch["data3"],
            org_sample_batch["data3"],
        )
    )


def test_AggregateDataAdapter_concat(sample_batch):
    # Define the aggregation type and batch aggregation definition

    cfg = load_config_from_path(
        file_path=Path("D4L-Hackaton")
        / "tests"
        / "utils"
        / "dummy_cfg-AggregateDataAdapter.yaml"
    )

    # Create an instance of the AggregateDataAdapter
    adapter = AggregateDataAdapter(cfg=cfg)

    org_sample_batch = sample_batch.copy()

    # Perform forward pass
    output_batch = adapter.forward(batch=sample_batch)["batch"]

    # Check if the aggregated data is present in the output batch
    assert "aggregated_data" in output_batch
    assert "aggregated_data2" in output_batch

    # Check if the data is aggregated correctly
    assert torch.all(
        torch.eq(
            output_batch["aggregated_data"],
            torch.cat([org_sample_batch["data1"], org_sample_batch["data2"]], dim=1),
        )
    )
    assert torch.all(
        torch.eq(
            output_batch["aggregated_data2"],
            org_sample_batch["data3"],
        )
    )


def test_AggregateDataAdapter_sum(sample_batch):
    # Define the aggregation type and batch aggregation definition

    cfg = load_config_from_path(
        file_path=Path("D4L-Hackaton")
        / "tests"
        / "utils"
        / "dummy_cfg-AggregateDataAdapter2.yaml"
    )

    # Create an instance of the AggregateDataAdapter
    adapter = AggregateDataAdapter(cfg=cfg)

    org_sample_batch = sample_batch.copy()

    # Perform forward pass
    output_batch = adapter.forward(batch=sample_batch)["batch"]

    # Check if the aggregated data is present in the output batch
    assert "aggregated_data" in output_batch
    assert "aggregated_data2" in output_batch

    # Check if the data is aggregated correctly
    assert torch.all(
        torch.eq(
            output_batch["aggregated_data"],
            org_sample_batch["data1"]
            + org_sample_batch["data2"]
            + org_sample_batch["data2"],
        ),
    )
    assert torch.all(
        torch.eq(
            output_batch["aggregated_data2"],
            org_sample_batch["data3"],
        )
    )


if __name__ == "__main__":
    pytest.main()
