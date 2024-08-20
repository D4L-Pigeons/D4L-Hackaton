from argparse import Namespace
from typing import Dict, List
import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from src.utils.common_types import ConfigStructure
from src.utils.config import validate_config_structure
from pathlib import Path


class hdf5SparseDataset(Dataset):
    r"""
    Used to create a dataset from an HDF5 file with sparse data.

        dataset_idxs (List[int]): List of indices to select from the dataset.
        cfg (Namespace): Configuration object containing the path and other parameters.


    Raises:
        AssertionError: If the indices in dataset_idxs are repeated.
        AssertionError: If the values in dataset_idxs are not within the range [0, max_possible_idx].
        NotImplementedError: If the remapping categories functionality is not implemented yet.

    """

    _config_structure: ConfigStructure = {
        "path": str,
        "rowsize": int,
        "obs": {
            "columns": [
                {
                    "org_name": str,
                    "new_name": str,
                    "remap_categories": bool,
                },
            ]
        },
    }

    def __init__(self, dataset_idxs: List[int], cfg: Namespace):
        super(hdf5SparseDataset).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self.cfg: Namespace = cfg
        self._dataset_idxs: np.ndarray = np.array(
            sorted(list(set(dataset_idxs)))
        )  # Unique values and sorting at the same time
        assert self._dataset_idxs.shape[0] == len(
            dataset_idxs
        ), "The indices in dataset_idxs should not repeat."
        self._len = self._dataset_idxs.shape[0]
        with h5py.File(Path(cfg.path), "r") as f:
            max_possible_idx = f["X"]["indptr"].shape[0] - 1
            assert (
                self._dataset_idxs[0] >= 0
                and self._dataset_idxs[-1]
                <= max_possible_idx  # Assuming sorted dataset_idxs
            ), f"Values in dataset_idxs should be within [0, {max_possible_idx}]."
            # BESIDES SAVING VAR REFERRING TO GENES ETC. SHOULD LEVERAGE SOME UNIFIED REMAPPING FOR DIFFERENT DATASETS
            self._var = f["var"]

    def __len__(self) -> int:
        return self._len

    def _map_idxs_to_dataset_idxs(self, idxs: List[int]) -> List[int]:
        return self._dataset_idxs[idxs]

    def __getitem__(self, index) -> Dict[str, Tensor]:
        return self.__getitems__([self._map_idxs_to_dataset_idxs(index)])

    def __getitems__(self, indices: List[int]) -> Dict[str, Tensor]:
        r"""
        Used by DataLoader to get items from the dataset.

        Args:
            indices (List[int]): The indices of the items to retrieve.

        Returns:
            Dict[str, Tensor]: A dictionary containing the data and additional information for the specified indices.

        """
        indices = self._map_idxs_to_dataset_idxs(
            sorted(indices)
        )  # Sorting as hdf5 selector requires increasing order of indexes.
        # Creating batch placeholder
        batch = {
            "data": torch.zeros(
                len(indices), self.cfg.rowsize
            ),  # Only nonzero entries will be filled
            **{
                col.new_name: torch.empty(len(indices), dtype=torch.long)
                for col in self.cfg.obs.columns
            },
        }
        with h5py.File(Path(self.cfg.path), "r") as f:
            for i, index in enumerate(indices):
                start, end = f["X"]["indptr"][[index, index + 1]]
                sparse_data = torch.tensor(f["X"]["data"][start:end])
                feature_indices = torch.tensor(
                    f["X"]["indices"][start:end], dtype=torch.int64
                )

                # Filling nonzero entries in a batch placeholder
                batch["data"][i, :] = torch.scatter(
                    input=torch.zeros(self.cfg.rowsize),
                    dim=0,
                    index=feature_indices,
                    src=sparse_data,
                )
            # TRY FASTER WITH STHG LIKE IN PREPROCESSING - REMINDER IN COMMENT BELOW
            # csr_matrix((data, indices, indptr), shape=shape)
            # batch["data"] = f["X/data"][indices]

            # Extracting data from .obs providing additional information about the cell. If remap_categories=True the remapping of codes is done to alter the numbering
            for obs_col in self.cfg.obs.columns:
                if obs_col.remap_categories:
                    raise NotImplementedError(
                        "The remapping categories functionality is not implemented yet."
                    )
                else:
                    batch[obs_col.new_name] = torch.tensor(
                        f["obs"][obs_col.org_name][indices], dtype=torch.long
                    )

        return batch
