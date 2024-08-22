import pytest
from argparse import Namespace
from src.utils.config import ConfigStructure, load_config_from_path
from src.utils.data.sc.dataset import hdf5SparseDataset
from pathlib import Path


def test_hdf5SparseDataset():
    # Test case 1: Valid dataset_idxs and cfg
    dataset_idxs = [0, 1, 2]
    cfg = load_config_from_path(
        file_path=Path("D4L-Hackaton")
        / "tests"
        / "utils"
        / "data"
        / "sc"
        / "dummy_cfgs"
        / "dummy_cfg-hdf5SparseDataset.yaml"
    )
    dataset = hdf5SparseDataset(dataset_idxs, cfg)
    assert len(dataset) == 3
    assert isinstance(dataset[0], dict)
    assert "cell_type" in dataset[0]
    assert "batch" in dataset[0]
    assert "site" in dataset[0]

    # Test case 2: Invalid dataset_idxs
    dataset_idxs = [0, 1, 1]  # Repeated indices
    with pytest.raises(AssertionError):
        dataset = hdf5SparseDataset(dataset_idxs, cfg)

    # Test case 3: Invalid cfg
    cfg = Namespace(
        path="path/to/hdf5_file.h5",
        rowsize=10,
        obs=Namespace(
            columns=[
                {"org_name": "column1", "new_name": "new_column1", "tocat": True},
                {"org_name": "column2", "new_name": "new_column2", "tocat": "invalid"},
            ]
        ),
    )
    with pytest.raises(AssertionError):
        dataset = hdf5SparseDataset(dataset_idxs, cfg)

    # Add more test cases for different scenarios and edge cases
