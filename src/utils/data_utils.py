import anndata as ad
import torch
from torch.utils.data import DataLoader

from utils.paths import ANNDATA_PATH


def load_anndata(mode: str, plus_iid_holdout: bool = False) -> ad.AnnData:
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

    filter_set = mode.split("+")  # ['train'] or ['test'] or ['train', 'test']

    if plus_iid_holdout:
        filter_set.append("iid_holdout")

    _data = ad.read_h5ad(ANNDATA_PATH)
    data = _data[_data.obs["is_train"].apply(lambda x: x in filter_set)]

    return data


def get_dataloader(data: ad.AnnData, batch_size: int):
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
    gex_indicator = (data.var["feature_types"] == "GEX").values
    torch.tensor(data.layers.toarray()[:gex_indicator], dtype=torch.float32)
    torch.tensor(data.layers.toarray()[:~gex_indicator], dtype=torch.float32)
