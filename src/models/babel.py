from argparse import Namespace
import anndata as ad
import pytorch_lightning as pl
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import Tensor
from anndata import AnnData
from sklearn.metrics import balanced_accuracy_score
from torch import Tensor
from typing import Dict, Optional, Tuple, Union

from utils.paths import LOGS_PATH
from utils.data_utils import get_dataloader_from_anndata
from models.ModelBase import ModelBase


class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(
                cfg.first_modality_dim + cfg.second_modality_dim, cfg.encoder_hidden_dim
            ),
            nn.BatchNorm1d(cfg.encoder_hidden_dim) if cfg.batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout_rate),
            nn.Linear(cfg.encoder_hidden_dim, cfg.encoder_hidden_dim),
            nn.BatchNorm1d(cfg.encoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout_rate),
            nn.Linear(cfg.encoder_hidden_dim, cfg.encoder_hidden_dim),
            nn.BatchNorm1d(cfg.encoder_hidden_dim) if cfg.batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout_rate),
            nn.Linear(cfg.encoder_hidden_dim, cfg.encoder_out_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.decoder_hidden_dim),
            nn.BatchNorm1d(cfg.decoder_hidden_dim) if cfg.batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout_rate),
            nn.Linear(cfg.decoder_hidden_dim, cfg.decoder_hidden_dim),
            nn.BatchNorm1d(cfg.decoder_hidden_dim) if cfg.batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout_rate),
            nn.Linear(cfg.decoder_hidden_dim, cfg.decoder_hidden_dim),
            nn.BatchNorm1d(cfg.decoder_hidden_dim) if cfg.batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout_rate),
            nn.Linear(
                cfg.decoder_hidden_dim, cfg.first_modality_dim + cfg.second_modality_dim
            ),
        )

    def forward(self, z):
        decoded = self.decoder(z)
        return decoded


class OmiModel(ModelBase):
    def __init__(self, cfg):
        super(OmiModel, self).__init__()
        self.cfg = cfg
        self.model = _OMIIVAE_IMPLEMENTATIONS[cfg.omivae_implementation](cfg)
        self.trainer = pl.Trainer(
            max_epochs=cfg.max_epochs,
            logger=pl.loggers.TensorBoardLogger(LOGS_PATH, name=cfg.model_name),
        )

    def train(self, train_anndata: AnnData, val_anndata: AnnData | None = None):
        train_loader = get_dataloader_from_anndata(
            data=train_anndata,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            first_modality_dim=self.cfg.first_modality_dim,
            second_modality_dim=self.cfg.second_modality_dim,
            include_class_labels=self.cfg.include_class_labels,
            target_hierarchy_level=self.cfg.target_hierarchy_level,
        )
        val_loader = (
            get_dataloader_from_anndata(
                data=val_anndata,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                first_modality_dim=self.cfg.first_modality_dim,
                second_modality_dim=self.cfg.second_modality_dim,
                include_class_labels=self.cfg.include_class_labels,
                target_hierarchy_level=self.cfg.target_hierarchy_level,
            )
            if val_anndata is not None
            else None
        )

        self.trainer.fit(
            model=self.model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

    def predict(self, anndata: AnnData) -> AnnData:
        pass

    def predict_proba(self, anndata: AnnData) -> Tensor:
        pass

    def save(self, file_path: str):
        save_path = file_path + ".ckpt"
        torch.save(self.model.state_dict(), save_path)
        return save_path

    def load(self, file_path: str):
        load_path = file_path + ".ckpt"
        self.model.load_state_dict(torch.load(load_path))

    def assert_cfg(self, cfg: Namespace) -> None:
        pass
