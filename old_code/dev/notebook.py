# from utils import data_utils
from argparse import Namespace
from pathlib import Path

# from sklearn.model_selection import train_test_split
from typing import Any, Dict, List, Optional, Tuple, Type

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import torch
from numpy import ndarray
from pandas import DataFrame
from scipy.sparse import csr_matrix
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.utils.paths import (  # EMBEDDINGS_PATH,; RAW_DATA_PATH,; PREPROCESSED_DATA_PATH,
    LOGS_PATH,
)


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
    train_dataloader: Type[DataLoader],
    val_dataloader: Type[DataLoader],
    cfg: Namespace,
) -> Any:
    r"""
    Builds and fits a model using the provided model constructor, configuration, training data, and validation data.

    Args:
        model_constructor (Type[Any]): The model constructor class.
        train_dataloader (Type[DataLoader]): The training dataloader.
        val_dataloader (Type[DataLoader]): The validation dataloader.
        cfg (Namespace): The configuration object containing the model's hyperparameters.

    Returns:
        Any: The trained model.
    """

    model = model_constructor(cfg)
    trainer = pl.Trainer(
        max_epochs=cfg.training.n_epochs,
        logger=pl.loggers.TensorBoardLogger(LOGS_PATH, name=cfg.model.model_name),
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    return model