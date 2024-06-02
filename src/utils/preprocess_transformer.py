import random

import numpy as np
import torch
from torch import Tensor

from utils.data_utils import get_dataset_from_anndata, load_anndata
from utils.paths import TRANSFORMER_DATA_PATH


def split_nonzero_zero_with_medians(data: Tensor):
    all_medians = torch.median(data, dim=0).values  # po wymiarze markerow
    assert all_medians.shape[0] == data.shape[1]
    nonzero_mask = all_medians != 0
    zero_mask = all_medians == 0

    medians_nonzero = all_medians[nonzero_mask]
    data_nonzero = data[:, nonzero_mask]
    assert data_nonzero.shape[0] == data.shape[0]
    data_zero = data[:, zero_mask]
    assert data_zero.shape[0] == data.shape[0]
    print("print number of markers with median zero:", data_zero.shape[1])
    print("print number of markers with median nonzero:", data_nonzero.shape[1])
    return medians_nonzero, data_nonzero, data_zero


def get_top_nonzeros(medians_nonzero, data_nonzero, n: int):
    top_n_indices = torch.argsort(medians_nonzero, descending=True)[:n]
    return data_nonzero[:, top_n_indices]


def get_random_zeros(data_zero, k: int):
    num_cols = data_zero.shape[1]
    random_indices = random.sample(range(num_cols), k)
    return data_zero[:, random_indices]


def select_top_n_and_random_k(data: Tensor, n: int, k: int):
    medians_nonzero, data_nonzero, data_zero = split_nonzero_zero_with_medians(data)

    top_n_data = get_top_nonzeros(medians_nonzero, data_nonzero, n)
    random_k_data = get_random_zeros(data_zero, k)

    combined_data = torch.cat((top_n_data, random_k_data), dim=1)
    print("Shape of combined data: ", combined_data.shape)
    return combined_data


def get_and_save_markers_subset(n: int = 256, k: int = 50, mode: str = "train"):
    data = load_anndata(mode=mode)
    gex_indicator = (data.var["feature_types"] == "GEX").values

    first_modality = torch.tensor(
        data.layers["counts"].toarray()[:, gex_indicator],
        dtype=torch.float32,
    )

    markers_subset = select_top_n_and_random_k(first_modality, n=n, k=k)
    torch.save(markers_subset, TRANSFORMER_DATA_PATH)


def main():
    get_and_save_markers_subset()


if __name__ == "__main__":
    main()
