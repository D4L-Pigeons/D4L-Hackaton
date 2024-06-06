import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse
import statsmodels.api as sm
import torch
import torch.utils
from torch.utils.data import DataLoader, TensorDataset

from utils.add_hierarchies import add_second_hierarchy
from utils.paths import ANNDATA_PATH, PREPROCESSED_ANNDATA_PATH


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


def remove_batch_effect(_data: ad.AnnData):
    pass


def GEX_preprocessing(_data: ad.AnnData):
    sc.pp.log1p(_data)
    sc.pp.scale(_data)


def ADT_preprocessing(_data: ad.AnnData):
    sc.pp.scale(_data)
    # sc.experimental.pp.normalize_pearson_residuals(_data)
    return
    # consider the following operations
    if scipy.sparse.issparse(X):
        mean = X.mean(axis=0).A1
        std = np.sqrt(X.power(2).mean(axis=0).A1 - np.square(mean))
        X = (X.toarray() - mean) / std
    else:
        mean = X.mean(axis=0)
        std = np.sqrt(X.square().mean(axis=0) - np.square(mean))
        X = (X - mean) / std
    X = X.clip(-10, 10)
    return X


def preprocess_andata(batch_effect: bool, normalize: bool) -> ad.AnnData:
    if normalize and PREPROCESSED_ANNDATA_PATH.exists():
        print("Loading precomputed log1p...")
        return ad.read_h5ad(PREPROCESSED_ANNDATA_PATH)

    _data = ad.read_h5ad(ANNDATA_PATH).toarray()
    if normalize:
        GEX_preprocessing(_data[_data.var["feature_types"] == "GEX"])
        ADT_preprocessing(_data[_data.var["feature_types"] == "ADT"])
        _data.write(filename=PREPROCESSED_ANNDATA_PATH)
    return _data


def load_anndata(
    mode: str,
    plus_iid_holdout: bool = False,
    normalize: bool = True,
    batch_effect: bool = True,
    add_hierarchy: bool = True,
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
    assert isinstance(
        normalize, bool
    ), f"normalize must be a boolean, got {normalize} instead."
    assert isinstance(
        batch_effect, bool
    ), f"batch_effect must be a boolean, got {batch_effect} instead."
    assert isinstance(
        add_hierarchy, bool
    ), f"batch_effect must be a boolean, got {add_hierarchy} instead."
    filter_set = mode.split("+")  # ['train'] or ['test'] or ['train', 'test']

    if plus_iid_holdout:
        filter_set.append("iid_holdout")

    # Read and normalize
    _data = preprocess_andata(batch_effect, normalize)

    if add_hierarchy:
        _data = add_second_hierarchy()

    data = _data[_data.obs["is_train"].apply(lambda x: x in filter_set)]

    return data


def get_dataset_from_anndata(
    data: ad.AnnData,
    first_modality_dim: int = 13953,
    second_modality_dim: int = 134,
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
    gex_indicator = (data.var["feature_types"] == "GEX").values
    assert gex_indicator.sum() >= first_modality_dim, (
        f"first_modality_dim must be less than or equal to the number of GEX features, "
        f"got {first_modality_dim} and {gex_indicator.sum()} instead."
    )
    assert (~gex_indicator).sum() >= second_modality_dim, (
        f"second_modality_dim must be less than or equal to the number of ADT features, "
        f"got {second_modality_dim} and {(~gex_indicator).sum()} instead."
    )
    first_modality = torch.tensor(
        data.layers["counts"][:, gex_indicator][:, :first_modality_dim],
        dtype=torch.float32,
    )
    second_modality = torch.tensor(
        data.layers["counts"][:, ~gex_indicator][:, :second_modality_dim],
        dtype=torch.float32,
    )
    if include_class_labels:
        labels = torch.tensor(data.obs["cell_type"].cat.codes.values, dtype=torch.long)
        dataset = TensorDataset(first_modality, second_modality, labels)
    else:
        dataset = TensorDataset(first_modality, second_modality)

    return dataset


def get_dataloader_from_anndata(
    data: ad.AnnData,
    first_modality_dim: int,
    second_modality_dim: int,
    batch_size: int,
    shuffle: bool = True,
    include_class_labels: bool = False,
) -> TensorDataset | DataLoader:
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


# def pearson_residuals_transform
