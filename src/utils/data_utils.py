from types import SimpleNamespace
from typing import Dict

import anndata as ad
import scanpy as sc
import torch
import torch.utils
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict
from types import SimpleNamespace
import os
from utils.add_hierarchies import add_second_hierarchy
from utils.paths import (
    RAW_ANNDATA_PATH,
    PREPROCESSED_ANNDATA_PATH,
    PREPROCESSED_DATA_PATH,
)


class Modality:
    GEX = "GEX"
    ADT = "ADT"


def _preprocess_modality(_data: ad.AnnData, modality_type: Modality):
    if modality_type == Modality.GEX:
        sc.pp.log1p(_data)
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
        _preprocess_modality(_data.X[:, gex_indicator], Modality.GEX)
        _preprocess_modality(_data.X[:, ~gex_indicator], Modality.ADT)
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
    assert isinstance(
        normalize, bool
    ), f"normalize must be a boolean, got {normalize} instead."
    assert isinstance(
        remove_batch_effect, bool
    ), f"remove_batch_effect must be a boolean, got {remove_batch_effect} instead."
    assert isinstance(
        target_hierarchy_level, int
    ), f"target_hierarchy_level must be a int, got {target_hierarchy_level} instead."
    assert target_hierarchy_level in [
        -1,
        -2,
    ], f"target_hierarchy_level must be in [-1, -2], got {target_hierarchy_level} instead."
    filter_set = mode.split("+")  # ['train'] or ['test'] or ['train', 'test']

    if plus_iid_holdout:
        filter_set.append("iid_holdout")

    # Read and normalize
    _data = ad.read_h5ad(RAW_ANNDATA_PATH)
    if normalize:
        sc.pp.log1p(_data)
    # _data = _preprocess_anndata(remove_batch_effect, normalize)

    if target_hierarchy_level == -2:
        _data = add_second_hierarchy(_data)

    if preload_subsample_frac is not None:
        print(f"Subsampling anndata with fraction {preload_subsample_frac}...")
        sc.pp.subsample(_data, fraction=preload_subsample_frac)

    data = _data[_data.obs["is_train"].apply(lambda x: x in filter_set)]

    return data


def get_modality_data_from_anndata(
    data: ad.AnnData, modality_cfg: SimpleNamespace
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
    assert modality_cfg.modality_name in [
        Modality.GEX,
        Modality.ADT,
    ], f"modality must be one of [{Modality.GEX}, {Modality.ADT}], got {modality_cfg.modality_name} instead."
    modality_indicator = (
        data.var["feature_types"] == modality_cfg.modality_name
    ).values
    assert modality_indicator.sum() >= modality_cfg.dim, (
        f"modality dim must be less than or equal to the number of {modality_cfg.modality_name} features, "
        f"got {modality_cfg.dim} and {modality_indicator.sum()} instead."
    )
    modality_data = torch.tensor(
        data[:, modality_indicator][:, : modality_cfg.dim].X.toarray(),
        dtype=torch.float32,
    )
    if torch.isnan(modality_data).any():
        print(
            f"{torch.isnan(modality_data).sum() / modality_data.numel() * 100}% of the values are nan in {modality_data} data."
        )
    else:
        print(f"There are no nan values in {modality_cfg.modality_name} data.")
    return modality_data


def get_data_dict_from_anndata(
    data: ad.AnnData,
    modalities_cfg: SimpleNamespace,
    include_class_labels: bool = False,
    target_hierarchy_level: int = -1,
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
        if target_hierarchy_level == -1:
            labels = torch.tensor(
                data.obs["cell_type"].cat.codes.values, dtype=torch.long
            )
        elif target_hierarchy_level == -2:
            labels = torch.tensor(
                data.obs["second_hierarchy"].cat.codes.values, dtype=torch.long
            )

        data_dict["labels"] = labels
    return data_dict


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
    modalities_cfg = SimpleNamespace(
        gex=SimpleNamespace(modality_name=Modality.GEX, dim=first_modality_dim),
        adt=SimpleNamespace(modality_name=Modality.ADT, dim=second_modality_dim),
    )
    data_dict = get_data_dict_from_anndata(
        data, modalities_cfg, include_class_labels, target_hierarchy_level
    )
    data_list = [data_dict["gex"], data_dict["adt"]]
    if include_class_labels:
        data_list.append(data_dict["labels"])
    dataset = TensorDataset(*data_list)
    return dataset


def get_dataloader_from_anndata(
    data: ad.AnnData,
    batch_size: int,
    shuffle: bool = True,
    first_modality_dim: int = 13953,
    second_modality_dim: int = 134,
    include_class_labels: bool = True,
    target_hierarchy_level: int = -1,
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


def get_dataloader_dict_from_anndata(
    data: ad.AnnData,
    cfg: SimpleNamespace,
    train: bool = True,
) -> Dict[str, DataLoader]:
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
    modalities_cfg = cfg.modalities
    data_dict = get_data_dict_from_anndata(
        data, modalities_cfg, cfg.include_class_labels, cfg.target_hierarchy_level
    )
    if train:
        batch_size = cfg.batch_size
    else:
        batch_size = cfg.predict_batch_size

    dataloader_dict = {
        cfg_modality_name: DataLoader(
            (
                TensorDataset(data_dict[cfg_modality_name], data_dict["labels"])
                if cfg.include_class_labels
                else TensorDataset(data_dict[cfg_modality_name])
            ),
            batch_size=batch_size,
            shuffle=train,
        )
        for cfg_modality_name in vars(cfg.modalities).keys()
    }
    return dataloader_dict


# def pearson_residuals_transform
