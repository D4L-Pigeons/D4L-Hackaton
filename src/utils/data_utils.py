import anndata as ad
import scanpy as sc
import torch
import torch.utils
from torch.utils.data import DataLoader, TensorDataset
import os

from utils.add_hierarchies import add_second_hierarchy
from utils.paths import (
    PREPROCESSED_ANNDATA_PATH,
    RAW_ANNDATA_PATH,
    PREPROCESSED_DATA_PATH,
)


def _GEX_preprocessing(_data: ad.AnnData):
    sc.pp.log1p(_data)
    sc.pp.scale(_data)


def _ADT_preprocessing(_data: ad.AnnData):
    sc.pp.scale(_data)
    # sc.experimental.pp.normalize_pearson_residuals(_data)

    # TODO: consider the following operations
    # if scipy.sparse.issparse(X):
    #     mean = X.mean(axis=0).A1
    #     std = np.sqrt(X.power(2).mean(axis=0).A1 - np.square(mean))
    #     X = (X.toarray() - mean) / std
    # else:
    #     mean = X.mean(axis=0)
    #     std = np.sqrt(X.square().mean(axis=0) - np.square(mean))
    #     X = (X - mean) / std
    # X = X.clip(-10, 10)


def _preprocess_anndata(remove_batch_effect: bool, normalize: bool) -> ad.AnnData:
    if normalize and PREPROCESSED_ANNDATA_PATH.exists():
        print("Loading preprocessed data...")
        return ad.read_h5ad(PREPROCESSED_ANNDATA_PATH)

    _data = ad.read_h5ad(RAW_ANNDATA_PATH)

    if normalize:
        gex_indicator = (_data.var["feature_types"] == "GEX").values
        _GEX_preprocessing(_data.layers["counts"][:, gex_indicator])
        _ADT_preprocessing(_data.layers["counts"][:, ~gex_indicator])
        if remove_batch_effect:  # Ensure data selection if needed
            sc.external.pp.bbknn(
                _data[:, gex_indicator], batch_key="Site", use_rep="GEX_X_pca"
            )
            sc.external.pp.bbknn(
                _data[:, ~gex_indicator], batch_key="Site", use_rep="ADT_X_pca"
            )
        if not os.path.exists(PREPROCESSED_DATA_PATH):
            os.makedirs(PREPROCESSED_DATA_PATH)
        _data.write(filename=PREPROCESSED_ANNDATA_PATH)
    return _data


def load_anndata(
    mode: str,
    plus_iid_holdout: bool = False,
    normalize: bool = True,
    remove_batch_effect: bool = True,
    target_hierarchy_level: int = -1,
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
        remove_batch_effect, bool
    ), f"remove_batch_effect must be a boolean, got {remove_batch_effect} instead."
    assert isinstance(
        target_hierarchy_level, int
    ), f"add_hierarchy must be a int, got {target_hierarchy_level} instead."
    assert target_hierarchy_level in [
        -1,
        -2,
    ], f"target_hierarchy_level must in {[-1, -2]}, got {target_hierarchy_level} instead"
    filter_set = mode.split("+")  # ['train'] or ['test'] or ['train', 'test']

    if plus_iid_holdout:
        filter_set.append("iid_holdout")

    # Read and normalize
    _data = _preprocess_anndata(remove_batch_effect, normalize)

    if target_hierarchy_level == -2:
        _data = add_second_hierarchy(_data)

    data = _data[_data.obs["is_train"].apply(lambda x: x in filter_set)]

    return data


def get_dataset_from_anndata(
    data: ad.AnnData,
    first_modality_dim: int = 13953,
    second_modality_dim: int = 134,
    include_class_labels: bool = True,
    target_hierarchy_level: int = -1,
) -> TensorDataset:
    r"""
    Get a TensorDataset object for the given data.

    Arguments:
    data : anndata.AnnData
        The data to create a TensorDataset for.
    first_modality_dim, second_modality_dim : int
        Number of sizes of each modality.
    include_class_labels : bool
        Number of sizes of each modality.
    target_hierarchy_level : int
        What hierarchy to use when include_class_labels=True.
        If equal -1, "cell_type" will be used, and if -2, "second_hierarchy".

    Returns:
    dataset : torch.utils.data.TensorDataset
        The TensorDataset object for the given data.
    """
    gex_indicator = data.var["feature_types"] == "GEX"
    assert gex_indicator.sum() >= first_modality_dim, (
        f"first_modality_dim must be less than or equal to the number of GEX features, "
        f"got {first_modality_dim} and {gex_indicator.sum()} instead."
    )
    assert (~gex_indicator).sum() >= second_modality_dim, (
        f"second_modality_dim must be less than or equal to the number of ADT features, "
        f"got {second_modality_dim} and {(~gex_indicator).sum()} instead."
    )
    first_modality = torch.tensor(
        data.layers["counts"].toarray()[:, gex_indicator][:, :first_modality_dim],
        dtype=torch.float32,
    )
    second_modality = torch.tensor(
        data.layers["counts"].toarray()[:, ~gex_indicator][:, :second_modality_dim],
        dtype=torch.float32,
    )
    if include_class_labels:
        implemented_hierarchy_levels_to_target_names = {
            -1: "cell_type",
            -2: "second_hierarchy",
        }
        assert (
            target_hierarchy_level
            in implemented_hierarchy_levels_to_target_names.keys()
        ), f"target_hierarchy_level must be one of {implemented_hierarchy_levels_to_target_names.keys()}, got {target_hierarchy_level} instead."
        target_name = implemented_hierarchy_levels_to_target_names[
            target_hierarchy_level
        ]
        labels = torch.tensor(data.obs[target_name].cat.codes.values, dtype=torch.long)
        dataset = TensorDataset(first_modality, second_modality, labels)
    else:
        dataset = TensorDataset(first_modality, second_modality)

    return dataset


def get_dataloader_from_anndata(
    data: ad.AnnData,
    batch_size: int,
    shuffle: bool = True,
    first_modality_dim: int = 13953,
    second_modality_dim: int = 134,
    include_class_labels: bool = True,
    target_hierarchy_level: int = -1,
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
        The DataLoader object for the given data starting with the GEX data.
    """
    dataset = get_dataset_from_anndata(
        data,
        first_modality_dim,
        second_modality_dim,
        include_class_labels,
        target_hierarchy_level,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader
