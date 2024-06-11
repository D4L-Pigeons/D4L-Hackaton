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
