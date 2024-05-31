from abc import ABC, abstractmethod

import anndata as ad
import numpy as np
from anndata import AnnData
from torch import Tensor

<<<<<<< HEAD
=======
from abc import ABC, abstractmethod
from argparse import Namespace

>>>>>>> main


class ModelBase(ABC):
    @abstractmethod
    def train(self, data: AnnData) -> None:
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
        assert hasattr(cfg, "first_modality_dim"), AttributeError(
            'cfg does not have the attribute "first_modality_dim"'
        )
        assert hasattr(cfg, "second_modality_dim"), AttributeError(
            'cfg does not have the attribute "second_modality_dim"'
        )
        assert hasattr(cfg, "hidden_dim"), AttributeError(
            'cfg does not have the attribute "hidden_dim"'
        )
        assert hasattr(cfg, "latent_dim"), AttributeError(
            'cfg does not have the attribute "latent_dim"'
        )
        assert hasattr(cfg, "batch_size"), AttributeError(
            'cfg does not have the attribute "batch_size"'
        )
        assert hasattr(cfg, "batch_norm"), AttributeError(
            'cfg does not have the attribute "batch_norm"'
        )

    @abstractmethod
    def assert_cfg(self, cfg: Namespace) -> None:
        pass
