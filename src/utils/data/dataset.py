from torch.utils.data import Dataset
from argparse import Namespace
from typing import Dict, List
import torch
from torch import Tensor
import h5py
import numpy as np


class hdf5SparseDataset(Dataset):
    r"""
    A dataset class for loading data from an HDF5 file in a sparse format.
    Requires specification of datast_idxs inducing a dataset as a subset of the whole data file.

    Args:
        cfg (Namespace): The configuration object containing the path to the HDF5 file.

    Attributes:
        cfg (Namespace): The configuration object containing the path to the HDF5 file.

    """

    def __init__(self, cfg: Namespace, dataset_idxs: List[int]):
        r"""
        Initialize the hdf5SparseDataset object.

        Args:
            cfg (Namespace): The configuration object.
            dataset_idxs (List[int]): The list of subset indices defining a dataset.

        Raises:
            AssertionError: If the indices in dataset_idxs are repeated.
            AssertionError: If values in dataset_idxs are not within [0, max_possible_idx].

        """
        super(hdf5SparseDataset).__init__()
        self.cfg = cfg
        self._dataset_idxs = np.array(
            list(set(dataset_idxs))
        )  # Unique values and sorting at the same time
        assert self._dataset_idxs.shape[0] == len(
            dataset_idxs
        ), "The indices in dataset_idxs should not repeat."
        self._len = self._dataset_idxs.shape[0]
        with h5py.File(cfg.path, "r") as f:
            max_possible_idx = f["X"]["indptr"].shape[0] - 1
            assert (
                self._dataset_idxs[0] >= 0
                and self._dataset_idxs[-1]
                <= max_possible_idx  # Assuming sorted dataset_idxs
            ), f"Values in dataset_idxs should be within [0, {max_possible_idx}]."
            self._var = f["var"]

    def __len__(self) -> int:
        return self._len

    def _map_idxs_to_dataset_idxs(self, idxs: List[int]) -> List[int]:
        return self._dataset_idxs[idxs]

    def __getitem__(self, index) -> Dict[str, Tensor]:
        return self.__getitems__([self._map_idxs_to_dataset_idxs([index])])

    def __getitems__(self, indices: List[int]) -> Dict[str, Tensor]:
        r"""
        Used by DataLoader to get items from the dataset.

        Args:
            indices (List[int]): The indices of the items to retrieve.

        Returns:
            Dict[str, Tensor]: A dictionary containing the data and additional information for the specified indices.

        """
        indices = self._map_idxs_to_dataset_idxs(indices)
        # Creating batch placeholder
        batch = {
            "data": torch.zeros(
                len(indices), self.cfg.rowsize
            ),  # Only nonzero entries will be filled
            **{
                col["new_name"]: torch.empty(len(indices), dtype=torch.long)
                for col in self.cfg.obs.columns
            },
        }
        with h5py.File(self.cfg.path, "r") as f:
            for i, index in enumerate(indices):
                start, end = f["X"]["indptr"][[index, index + 1]]
                sparse_data = torch.tensor(f["X"]["data"][start:end])
                feature_indices = torch.tensor(
                    f["X"]["indices"][start:end], dtype=torch.int64
                )

                # Filling the nonzero entries
                batch["data"][i, :] = torch.scatter(
                    input=torch.zeros(self.cfg.rowsize),
                    dim=0,
                    index=feature_indices,
                    src=sparse_data,
                )

                # Extracting data from .obs providing additional information about the cell
                for obs_col in self.cfg.obs.columns:
                    batch[obs_col["new_name"]][i] = (
                        torch.tensor(
                            f["obs"][obs_col["org_name"]][index],
                            dtype=torch.long,
                        )
                        if obs_col["tocat"]
                        else torch.tensor(
                            f["obs"][obs_col["org_name"]][index].apply_along_axis(
                                func1d=lambda x: f["obs"]["__categories"][
                                    obs_col["org_name"]
                                ]
                            ),
                            dtype=torch.float,
                        )
                    )

        return batch
