import anndata as ad
import numpy as np
import scanpy as sc
import statsmodels.api as sm
import torch
import torch.utils
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict
from types import SimpleNamespace
from utils.paths import (
    ANNDATA_PATH,
    LOG1P_ANNDATA_PATH,
    PEARSON_RESIDUALS_ANNDATA_PATH,
    STD_ANNDATA_PATH,
)

class Modality:
    GEX = "GEX"
    ADT = "ADT"

def fit_negative_binomial(counts):
    """
    Fit a negative binomial model to each gene and estimate the dispersion parameter.

    Parameters:
    counts (numpy.ndarray): The count matrix with genes as rows and cells as columns.

    Returns:
    tuple: A tuple containing the mean counts and dispersion parameters for each gene.
    """
    n_genes, n_cells = counts.shape
    mean_counts = np.mean(counts, axis=1)
    dispersions = np.zeros(n_genes)

    for i in range(n_genes):
        y = counts[i, :]
        X = np.ones((n_cells, 1))  # Design matrix with intercept only

        # Fit the negative binomial model
        model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
        results = model.fit()

        # Extract the dispersion parameter
        dispersions[i] = results.scale

    return mean_counts, dispersions


# mean_counts, dispersions = fit_negative_binomial(counts)


# def memory_aware_pearson_residuals(data: ad.AnnData) -> ad.AnnData:
#     r"""
#     Compute Pearson residuals in a memory-aware way.

#     Arguments:
#     data : anndata.AnnData
#         The data to compute Pearson residuals for.

#     Returns:
#     data : anndata.AnnData
#         The data with Pearson residuals computed.
#     """
#     # Compute Pearson residuals in an
#     # n_cols = data.shape[1]


def load_anndata(
    mode: str,
    plus_iid_holdout: bool = False,
    preprocessing: str | None = "pearson_residuals",
    preload_subsample_frac: float = 1.0,
) -> ad.AnnData:
    r"""
    Load the full anndata object for the specified mode.

    Arguments:
    mode : str
        The mode to load the anndata object for. Must be one of ['train', 'test', 'train+test'].
    plus_iid_holdout : bool
        Whether to include the iid_holdout data in the anndata object.

    Returns:
    data : anndata.AnnData
        The full anndata object for the specified mode.
    """
    assert mode in [
        "train",
        "test",
        "train+test",
    ], f"mode must be one of ['train', 'test', 'train+test'], got {mode} instead."
    assert isinstance(
        plus_iid_holdout, bool
    ), f"plus_iid_holdout must be a boolean, got {plus_iid_holdout} instead."
    assert preprocessing in [
        None,
        "log1p",
        "standardize",
        "pearson_residuals",
    ], f"preprocessing must be one of [None, 'log1p', 'standardize', 'pearson_residuals'], got {preprocessing} instead."
    filter_set = mode.split("+")  # ['train'] or ['test'] or ['train', 'test']

    if plus_iid_holdout:
        filter_set.append("iid_holdout")

    _data = ad.read_h5ad(ANNDATA_PATH)
    if preload_subsample_frac is not None:
        print(f"Subsampling anndata with fraction {preload_subsample_frac}...")
        sc.pp.subsample(_data, fraction=preload_subsample_frac)
    if preprocessing == "log1p":
        if not LOG1P_ANNDATA_PATH.exists():
            print("Preprocessing with log1p...")
            sc.pp.log1p(_data)
            _data.write(filename=LOG1P_ANNDATA_PATH)
        else:
            print("Loading precomputed log1p...")
            _data = ad.read_h5ad(LOG1P_ANNDATA_PATH)
    elif preprocessing == "standardize":
        if not STD_ANNDATA_PATH.exists():
            print("Preprocessing with standardize...")
            sc.pp.scale(_data)
            _data.write(filename=STD_ANNDATA_PATH)
        else:
            print("Loading precomputed standardize...")
            _data = ad.read_h5ad(STD_ANNDATA_PATH)
    # if preprocessing == "pearson_residuals":
    #     if not PEARSON_RESIDUALS_ANNDATA_PATH.exists():
    #         print("Normalizing Pearson residuals...")
    #         sc.experimental.pp.normalize_pearson_residuals(_data)
    #         _data.write(filename=PEARSON_RESIDUALS_ANNDATA_PATH)
    #     else:
    #         print("Loading precomputed Pearson residuals...")
    #         _data = ad.read_h5ad(PEARSON_RESIDUALS_ANNDATA_PATH)
    data = _data[_data.obs["is_train"].apply(lambda x: x in filter_set)]

    return data


def get_modality_data_from_anndata(
    data: ad.AnnData,
    modality_cfg: SimpleNamespace
) -> torch.Tensor:
    r"""
    Get a TensorDataset object for the given modality.

    Arguments:
    data : anndata.AnnData
        The data to create a TensorDataset for.
    modality_cfg : types.SimpleNamespace
        The configuration for the modality.

    Returns:
    modality_data : torch.Tensor
        The TensorDataset object for the given modality.
    """
    assert modality_cfg.modality_name in [Modality.GEX, Modality.ADT], (
        f"modality must be one of [{Modality.GEX}, {Modality.ADT}], got {modality_cfg.modality_name} instead."
    )
    modality_indicator = (data.var["feature_types"] == modality_cfg.modality_name)
    modality_indicator = modality_indicator.values
    assert modality_indicator.sum() >= modality_cfg.dim, (
        f"modality dim must be less than or equal to the number of {modality_cfg.modality_name} features, "
        f"got {modality_cfg.dim} and {modality_indicator.sum()} instead."
    )
    modality_data = torch.tensor(
        data.layers["counts"].toarray()[:, modality_indicator][:, :modality_cfg.dim],
        dtype=torch.float32,
    )
    if torch.isnan(modality_data).any():
        print(
            f"{torch.isnan(modality_data).sum() / modality_data.numel() * 100}% of the values are nan in {modality} data."
        )
    else:
        print(f"There are no nan values in {modality_cfg.modality_name} data.")
    return modality_data

def get_data_dict_from_anndata(
    data: ad.AnnData,
    modalities_cfg: SimpleNamespace,
    include_class_labels: bool = False,
    ) -> Dict[str, torch.Tensor]:
    r"""
    Get a TensorDataset object for the given data.

    Arguments:
    data : anndata.AnnData
        The data to create a TensorDataset for.

    Returns:
    data_dict : dict of torch.Tensor
    """
    data_dict = {
        cfg_name: get_modality_data_from_anndata(data, modality_cfg)
        for cfg_name, modality_cfg in vars(modalities_cfg).items()
    }
    if include_class_labels:
        labels = torch.tensor(data.obs["cell_type"].cat.codes.values, dtype=torch.long)
        data_dict["labels"] = labels
    return data_dict

def get_dataset_from_anndata(
    data: ad.AnnData,
    first_modality_dim: int,
    second_modality_dim: int,
    include_class_labels: bool = False,
) -> TensorDataset:
    r"""
    Get a TensorDataset object for the given data.

    Arguments:
    data : anndata.AnnData
        The data to create a TensorDataset for.

    Returns:
    dataset : torch.utils.data.TensorDataset
        The TensorDataset object for the given data.
    """
    modalities_cfg = SimpleNamespace(
        gex=SimpleNamespace(modality_name=Modality.GEX, dim=first_modality_dim),
        adt=SimpleNamespace(modality_name=Modality.ADT, dim=second_modality_dim),
    )
    data_dict = get_data_dict_from_anndata(data, modalities_cfg, include_class_labels)
    data_list = [data_dict["gex"], data_dict["adt"]]
    if include_class_labels:
        data_list.append(data_dict["labels"])
    dataset = TensorDataset(
        *data_list
    )
    return dataset

def get_dataloader_from_anndata(
    data: ad.AnnData,
    first_modality_dim: int,
    second_modality_dim: int,
    batch_size: int,
    shuffle: bool = True,
    include_class_labels: bool = False,
) -> DataLoader:
    r"""
    Get a DataLoader object for the given data.

    Arguments:
    data : anndata.AnnData
        The data to create a DataLoader for.
    batch_size : int
        The batch size to use for the DataLoader.

    Returns:
    dataloader : torch.utils.data.DataLoader
        The DataLoader object for the given data. With the GEX data first.
    """
    dataset = get_dataset_from_anndata(
        data, first_modality_dim, second_modality_dim, include_class_labels
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def get_dataloader_dict_from_anndata(
    data: ad.AnnData,
    modalities_cfg: SimpleNamespace,
    shuffle: bool = True,
) -> DataLoader:
    r"""
    Get a DataLoader object for the given data.

    Arguments:
    data : anndata.AnnData
        The data to create a DataLoader for.
    batch_size : int
        The batch size to use for the DataLoader.

    Returns:
    dataloader : torch.utils.data.DataLoader
        The DataLoader object for the given data. With the GEX data first.
    """
    data_dict = get_data_dict_from_anndata(data, modalities_cfg)
    dataloader_dict = {
        cfg_name: DataLoader(
            TensorDataset(data), batch_size=vars(modalities_cfg)[cfg_name].batch_size, shuffle=shuffle
        )
        for cfg_name, data in data_dict.items()
    }
    return dataloader_dict


# def pearson_residuals_transform
