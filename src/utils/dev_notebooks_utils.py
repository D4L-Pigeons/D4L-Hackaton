import torch
from torch import Tensor
import pandas as pd
from pandas import DataFrame
import anndata as ad
import scanpy as sc
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Type, Any, Optional, Dict
from utils import data_utils
from argparse import Namespace
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import h5py
from utils.paths import (
    EMBEDDINGS_PATH,
    LOGS_PATH,
    RAW_DATA_PATH,
    PREPROCESSED_DATA_PATH,
)
from pathlib import Path
from numpy import ndarray
from scipy.sparse import csr_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import ndarray


def dict_to_namespace(d: Dict) -> Namespace:
    r"""
    Recursively converts a dictionary into a SimpleNamespace object.

    Args:
        d (Dict): The dictionary to be converted.

    Returns:
        Namespace: The converted Namespace object.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)
    return Namespace(**d)


# --- TRAINING ---


def build_and_fit_model(
    model_constructor: Type[Any],
    cfg: Namespace,
    train_tensor_data: Tensor,
    val_tensor_data: Tensor,
) -> Any:
    r"""
    Builds and fits a model using the provided model constructor, configuration, training data, and validation data.

    Args:
        model_constructor (Type[Any]): The constructor function for the model.
        cfg (Namespace): The configuration object containing the model's hyperparameters.
        train_tensor_data (Tensor): The training data tensor.
        val_tensor_data (Tensor): The validation data tensor.

    Returns:
        Any: The trained model.
    """
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


# --- EMBEDDINGS ---


def get_data_embeddings(
    tensor_dataset: TensorDataset, model, batch_size: int = 1
) -> Path:
    r"""
    Computes the embeddings of the data using the model and saves them to a HDF5 file.

    Args:
        tensor_dataset (TensorDataset): The input tensor dataset.
        model: The model used to compute the embeddings.
        batch_size (int, optional): The batch size for the data loader. Defaults to 1.

    Returns:
        Path: The path to the saved HDF5 file containing the embeddings.
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
            x.to(model.device)
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

    Args:
        tensor_dataset (TensorDataset): The input tensor dataset.
        model: The model used to compute the embeddings.
        batch_size (int, optional): The batch size for the data loader. Defaults to 1.

    Returns:
        Path: The path to the saved HDF5 file containing the embeddings.
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


# --- VISUALIZATIONS ---


def draw_umaps_pca(
    data: ndarray, df: DataFrame, n_comps: int = 20, n_neighbors: int = 10
) -> None:
    r"""
    Computes PCA and draws UMAPs for each column in the DataFrame.

    Parameters:
        data (ndarray): The input data array.
        df (DataFrame): The DataFrame containing the columns to visualize.
        n_comps (int, optional): The number of principal components to compute in PCA. Defaults to 20.
        n_neighbors (int, optional): The number of neighbors to consider in UMAP. Defaults to 10.
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


# --- METRICS ---


def __calc_entropy(preds_probs: ndarray) -> float:
    r"""
    Calculates the entropy of a distribution.

    Parameters:
    preds_probs (ndarray): The predicted probabilities of the distribution.

    Returns:
    float: The entropy value.

    """
    return -np.sum(preds_probs * np.log(preds_probs + 1e-10), axis=1).mean()


def calc_metrics_from_uniport(
    train_embeddings: ndarray,
    generalization_embeddings: ndarray,
    train_df: DataFrame,
    generalization_df: DataFrame,
    n_neighbors: int = 50,
    n_loc_samples: int = 100,
) -> DataFrame:
    r"""
    Calculates the metrics from the embeddings and the metadata.

    Args:
        train_embeddings (ndarray): Array of training embeddings.
        generalization_embeddings (ndarray): Array of generalization embeddings.
        train_df (DataFrame): DataFrame containing training metadata.
        generalization_df (DataFrame): DataFrame containing generalization metadata.
        n_neighbors (int, optional): Number of neighbors to consider for KNN classification. Defaults to 50.
        n_loc_samples (int, optional): Number of samples to select from generalization embeddings. Defaults to 100.

    Returns:
        DataFrame: DataFrame containing calculated metrics for each column in train_df.

    Note:
        AverageFOSCTTM is not calculated.
    """
    knc = KNeighborsClassifier(n_neighbors=n_neighbors)

    metrics = pd.DataFrame(
        columns=[
            "col_name",
            "AdjustedRandIndex",
            "NormalizedMutualInformation",
            "F1Score",
            "SilhouetteCoefficient",
            "MixingEntropyScore",
        ],
        index=train_df.columns,
    )
    sampled_region_indices = torch.randint(
        0, generalization_embeddings.shape[0], (n_loc_samples,)
    )
    for col_name in train_df.columns:
        knc.fit(train_embeddings, train_df[col_name])
        labels = generalization_df[col_name]
        preds = knc.predict(generalization_embeddings)
        preds_probs = knc.predict_proba(
            generalization_embeddings[sampled_region_indices, :]
        )
        metrics.loc[col_name] = [
            col_name,
            adjusted_rand_score(labels, preds),
            normalized_mutual_info_score(labels, preds),
            f1_score(labels, preds, average="weighted"),
            (silhouette_score(generalization_embeddings, labels) + 1)
            * 0.5,  # Range [0-1] instead of [-1, 1]
            __calc_entropy(preds_probs),
        ]

    return metrics
