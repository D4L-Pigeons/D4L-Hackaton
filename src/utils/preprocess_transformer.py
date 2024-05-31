import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.utils.data_utils import get_dataset_from_anndata, load_anndata
from utils.paths import ANNDATA_PATH, PEARSON_RESIDUALS_ANNDATA_PATH


def calculate_medians(data: TensorDataset):
    pass


def split_nonzero_zero_with_medians(data: TensorDataset):
    all_medians = calculate_medians(data)
    return medians_nonzero, data_nonzero, medians_zero, data_zero


def get_top_nonzeros(medians_nonzero, data_nonzero, n: int):
    pass


def get_random_zeros(medians_zero, data_zero, k: int):
    pass


def get_markers_subset(n: int = 256, k: int = 50, mode: str = "train"):
    data = load_anndata(mode=mode)
    get_dataset_from_anndata(data, first_modality_dim, second_modality_dim)
    (
        medians_nonzero,
        data_nonzero,
        medians_zero,
        data_zero,
    ) = split_nonzero_zero_with_medians(data)
    nonzeros = get_top_nonzeros(medians_nonzero, data_nonzero, n)
    zeros = get_random_zeros(medians_zero, data_zero, k)
    markers_subset = np.cat(nonzeros, zeros)
    return markers_subset
