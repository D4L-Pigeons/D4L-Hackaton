from abc import ABC, abstractmethod
from argparse import Namespace

import anndata as ad
import numpy as np
from anndata import AnnData
from torch import Tensor


class ModelBase(ABC):
    @abstractmethod
    def fit(self, data: AnnData) -> None:
        pass

    @abstractmethod
    def predict(self, data: AnnData) -> Tensor:
        pass

    @abstractmethod
    def predict_proba(self, data: AnnData) -> Tensor:
        pass

    @abstractmethod
    def save(self, file_path: str) -> None:
        pass

    @abstractmethod
    def load(self, file_path: str) -> None:
        pass

    @staticmethod
    def assert_cfg_general(cfg: Namespace) -> None:
        default_cfg = {
            "first_modality_dim": 13953,
            "second_modality_dim": 134,
            "latent_dim": 20,
            "batch_size": 128,
            "batch_norm": False,
            "include_class_labels": True,
        }

        # Use getattr with default values
        for attr, default_value in default_cfg.items():
            if not hasattr(cfg, attr):
                setattr(cfg, attr, default_value)
                print(f"{attr} set as {default_value}")

    @abstractmethod
    def assert_cfg(self, cfg: Namespace) -> None:
        pass
