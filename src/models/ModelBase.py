from abc import ABC, abstractmethod

import anndata as ad
import numpy as np
from torch import Tensor


class ModelBase(ABC):
    @abstractmethod
    def encode(self, data: ad.AnnData) -> Tensor:
        pass

    @abstractmethod
    def decode(self, data: np.ndarray) -> Tensor:
        pass

    @abstractmethod
    def get_decoder_jacobian(self, data: ad.AnnData) -> Tensor:
        pass

    @abstractmethod
    def train(self, data: ad.AnnData) -> None:
        pass

    @abstractmethod
    def predict(self, data: ad.AnnData) -> Tensor:
        pass

    @abstractmethod
    def predict_proba(self, data: ad.AnnData) -> Tensor:
        pass

    @abstractmethod
    def save(self, file_path: str) -> str:
        pass

    @abstractmethod
    def load(self, file_path: str) -> None:
        pass
