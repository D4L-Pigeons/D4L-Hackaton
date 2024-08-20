import torch
import pytorch_lightning as pl
from src.models.components.chain_model import ChainModel
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace
from src.utils.common_types import ConfigStructure, Batch, StructuredForwardOutput
from src.utils.config import validate_config_structure
from src.models.components.loss import get_loss_aggregator


class DenseAE(pl.LightningModule):
    _config_structure: ConfigStructure = {
        "chain": [Namespace],
        "reconstr_var_name": str,
        "loss_aggregator": str,
        "reconstr_loss": str,
    }

    def __init__(self, cfg: Namespace) -> None:
        super(DenseAE, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._reconstr_var_name: str = cfg.reconst
        self.model: ChainModel = ChainModel(cfg=cfg.chain)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        reconst_variable: torch.Tensor = batch[
            self._reconstr_var_name
        ].clone()  # Is this clone necessary?
        output: StructuredForwardOutput = self.model(batch)

        # output["batch"]
        # self.log_dict(output["losses"])
