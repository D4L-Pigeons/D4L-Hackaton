import torch
from torch import Tensor
import pandas as pd
from pandas import DataFrame
import anndata as ad
import scanpy as sc
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Type, Any
from utils import data_utils
from argparse import Namespace
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import h5py
from utils.paths import EMBEDDINGS_PATH, LOGS_PATH
from pathlib import Path
from numpy import ndarray


# Note: only preprosessing, which keeps 0 as 0 is allowed if we want 0 to mean no expression
def prepare_data_naive_mixing_version(
    val_frac: float,
    divide_by_nonzero_median: bool,
    df_columns: List[str],
    random_state: int = 0,
    return_raw: bool = False,
) -> (
    Tuple[Tuple[Tensor, Tensor], Tuple[DataFrame, DataFrame], Tuple[Tensor, Tensor]]
    | Tuple[Tuple[Tensor, Tensor], Tuple[DataFrame, DataFrame]]
):
    data = data_utils.load_anndata(mode="train", plus_iid_holdout=True, normalize=False)
    sc.pp.log1p(data)
    data = data[data.obs["is_train"].apply(lambda x: x in ["train", "iid_holdout"])]

    tensor_data = torch.tensor(data.X.toarray())
    raw = torch.tensor(data.layers["counts"].toarray())
    df = pd.DataFrame(data.obs[df_columns])
    del data
    if divide_by_nonzero_median:
        nonzero_median = tensor_data[tensor_data > 0].median(dim=0, keepdim=True).values
        tensor_data = tensor_data / nonzero_median
        del nonzero_median

    train_tensor_data, val_tensor_data, train_df, val_df, train_raw, val_raw = (
        train_test_split(
            tensor_data, df, raw, test_size=val_frac, random_state=random_state
        )
    )

    if return_raw:
        return (
            (train_tensor_data, val_tensor_data),
            (train_df, val_df),
            (train_raw, val_raw),
        )

    return (train_tensor_data, val_tensor_data), (train_df, val_df)


def draw_umaps_pca(
    data: ndarray, df: DataFrame, n_comps: int = 20, n_neighbors: int = 10
) -> None:
    r"""
    Computes pca and draws umaps for each column in df.
    """
    assert (
        n_comps <= data.shape[1]
    ), "n_comps cannot be greater than the number of features"
    ad_tmp = ad.AnnData(X=data, obs=df)
    sc.pp.pca(ad_tmp, n_comps=n_comps)
    sc.pp.neighbors(ad_tmp, n_neighbors=n_neighbors, use_rep="X_pca")
    sc.tl.umap(ad_tmp)

    for col_name in df.columns:
        sc.pl.umap(ad_tmp, color=col_name)

    for col_name in df.columns:
        sc.pl.pca(ad_tmp, color=col_name)


def dict_to_namespace(d):
    r"""
    Recursively converts a dictionary into a Namespace object.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)
    return Namespace(**d)


def get_data_embeddings(
    tensor_dataset: TensorDataset, model, latent_dim: int, batch_size: int = 1
) -> Path:
    r"""
    Computes the embeddings of the data using the model and saves them to an HDF5 file.
    """
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    # Open an HDF5 file
    with h5py.File(
        EMBEDDINGS_PATH / f"{model.cfg.model.model_name}-embeddings.h5", "w"
    ) as h5f:
        # Create a resizable dataset to store embeddings
        max_shape = (
            None,
            model.cfg.model.latent_dim,
        )  # None indicates that the dimension is resizable
        dset = h5f.create_dataset(
            "embeddings",
            shape=(0, model.cfg.model.latent_dim),
            maxshape=max_shape,
            dtype="float32",
        )

        for i, x in enumerate(dataloader):
            x = x[0]
            model.to(x.device)
            with torch.no_grad():
                encoded = model.encode(x)

            # Resize the dataset to accommodate new embeddings
            dset.resize(dset.shape[0] + encoded.shape[0], axis=0)
            # Write the new embeddings
            dset[-encoded.shape[0] :] = encoded.detach().cpu().numpy()

    return EMBEDDINGS_PATH / f"{model.cfg.model.model_name}-embeddings.h5"


def get_data_embeddings_transformer_version(
    tensor_dataset: TensorDataset, model, batch_size: int = 1
) -> Path:
    r"""
    Computes the embeddings of the data using the model and saves them to an HDF5 file.
    """
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    # Open an HDF5 file
    with h5py.File(
        EMBEDDINGS_PATH / f"{model.cfg.model.model_name}-embeddings.h5", "w"
    ) as h5f:
        # Create a resizable dataset to store embeddings
        max_shape = (
            None,
            model.cfg.model.latent_dim,
        )  # None indicates that the dimension is resizable
        dset = h5f.create_dataset(
            "mu_embeddings",
            shape=(0, model.cfg.model.latent_dim),
            maxshape=max_shape,
            dtype="float32",
        )

        for i, x in enumerate(dataloader):
            x = x[0]
            encoder_input_gene_idxs, aux_gene_idxs = model.gene_choice_module.choose(x)
            encoder_input_exprs_lvls = torch.gather(
                x, dim=1, index=encoder_input_gene_idxs
            )
            model.to(encoder_input_exprs_lvls.device)
            with torch.no_grad():
                _, mu, _ = model.encode(
                    encoder_input_gene_idxs, encoder_input_exprs_lvls
                )

            # Resize the dataset to accommodate new embeddings
            dset.resize(dset.shape[0] + mu.shape[0], axis=0)
            # Write the new embeddings
            dset[-mu.shape[0] :] = mu.detach().cpu().numpy()

    return EMBEDDINGS_PATH / f"{model.cfg.model.model_name}-embeddings.h5"


def build_and_fit_model(
    model_constructor: Type[Any],
    cfg: Namespace,
    train_tensor_data: Tensor,
    val_tensor_data: Tensor,
) -> Any:
    train_tensor_dataset = TensorDataset(train_tensor_data)
    train_loader = DataLoader(
        train_tensor_dataset, batch_size=cfg.training.batch_size, shuffle=True
    )
    val_tensor_dataset = TensorDataset(val_tensor_data)
    val_loader = DataLoader(
        val_tensor_dataset, batch_size=cfg.training.batch_size, shuffle=False
    )
    model = model_constructor(cfg)
    trainer = pl.Trainer(
        max_epochs=cfg.training.n_epochs,
        logger=pl.loggers.TensorBoardLogger(LOGS_PATH, name=cfg.model.model_name),
    )
    trainer.fit(model, train_loader, val_loader)
    return model
