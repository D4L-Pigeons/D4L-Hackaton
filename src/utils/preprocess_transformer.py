import numpy as np
import torch
from torch.utils.data import Tensor
import random

from src.utils.data_utils import get_dataset_from_anndata, load_anndata
from utils.paths import ANNDATA_PATH


def split_nonzero_zero_with_medians(data: Tensor):
    all_medians = torch.median(data, dim=0).values
    nonzero_mask = all_medians != 0
    zero_mask = all_medians == 0

    medians_nonzero = all_medians[nonzero_mask]
    data_nonzero = data[:, nonzero_mask]
    data_zero = data[:, zero_mask]
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
    return combined_data


def get_markers_subset(n: int = 256, k: int = 50, mode: str = "train"):
    data_anndata = load_anndata(mode=mode)
    data = get_dataset_from_anndata(
        data_anndata, first_modality_dim, second_modality_dim
    )
    medians_nonzero, data_nonzero, data_zero = select_top_n_and_random_k(data)
    nonzeros = get_top_nonzeros(medians_nonzero, data_nonzero, n)
    zeros = get_random_zeros(data_zero, k)
    markers_subset = np.cat(nonzeros, zeros)
    return markers_subset
