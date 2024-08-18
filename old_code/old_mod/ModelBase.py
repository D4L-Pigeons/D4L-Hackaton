from abc import ABC, abstractmethod

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
    def save(self, file_path: str) -> None:
        pass

    @abstractmethod
    def load(self, file_path: str) -> None:
        pass
