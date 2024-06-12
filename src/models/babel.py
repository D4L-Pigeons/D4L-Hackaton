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
from utils.data_utils import get_dataloader_dict_from_anndata
from models.ModelBase import ModelBase
from pytorch_lightning.utilities.combined_loader import CombinedLoader


class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(cfg.dim, cfg.encoder_hidden_dim),
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
            nn.Linear(cfg.decoder_hidden_dim, cfg.dim),
        )

    def forward(self, z):
        decoded = self.decoder(z)
        return decoded


class SingleModalityVAE(nn.Module):
    def __init__(self, cfg):
        super(SingleModalityVAE, self).__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.normal_dist = td.Normal(0, 1)

    def encode(self, x):
        mu, log_var = self.encoder(x).chunk(2, dim=-1)
        std = torch.exp(0.5 * log_var)
        eps = self.normal_dist.sample(std.size()).to(mu.device)
        z = mu + eps * std
        return z, mu, std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, std = self.encode(x)
        decoded = self.decode(z)
        return decoded, mu, std


class BabelVAE(pl.LightningModule):
    def __init__(self, cfg):
        super(BabelVAE, self).__init__()
        self.cfg = cfg
        self.model = nn.ModuleDict(
            {
                cfg_name: SingleModalityVAE(modality_cfg)
                for cfg_name, modality_cfg in vars(cfg.modalities).items()
            }
        )

        self.trainer = pl.Trainer(
            max_epochs=cfg.max_epochs,
            log_every_n_steps=cfg.log_every_n_steps,
            logger=pl.loggers.TensorBoardLogger(LOGS_PATH, name=cfg.model_name),
            callbacks=(
                [
                    pl.callbacks.EarlyStopping(
                        monitor="val_loss",
                        min_delta=cfg.min_delta,
                        patience=cfg.patience,
                        verbose=False,
                        mode="min",
                    )
                ]
                if cfg.early_stopping
                else []
            ),
        )

    def training_step(
        self, batch: Dict[str, Tuple[Tensor]]
    ) -> Tuple[Tensor, Dict[str, float]]:
        total_loss = 0.0
        full_losses_dict = {}
        for encoding_modality_name, encoding_model in self.model.items():
            for decoding_modality_name, decoding_model in self.model.items():
                (x_enc,) = batch[encoding_modality_name]
                (x_dec,) = batch[decoding_modality_name]
                encoded, mu, std = encoding_model.encode(x_enc)
                decoded = decoding_model.decode(encoded)
                kld_loss = self.kld_divergence(mu, std)
                recon_loss = F.mse_loss(decoded, x_dec)
                total_loss += (
                    self.cfg.recon_loss_coef * recon_loss
                    + self.cfg.kld_loss_coef * kld_loss
                )
                full_losses_dict[
                    f"{encoding_modality_name}_{decoding_modality_name}_kld"
                ] = kld_loss.detach().item()
                full_losses_dict[
                    f"{encoding_modality_name}_{decoding_modality_name}_recon"
                ] = recon_loss.detach().item()
        self.log_dict(
            full_losses_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return total_loss

    def kld_divergence(self, mu, std):
        return -0.5 * torch.sum(2 * torch.log(std) - std**2 - mu**2 + 1)

    def validation_step(self, batch: Dict[str, Tuple[Tensor]]) -> Dict[str, float]:
        full_losses_dict = {}
        for modality_name, model in self.model.items():
            losses_dict = model.validation_step(batch[modality_name])
            full_losses_dict.update(losses_dict)
        self.log_dict(
            full_losses_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

    def get_dataloader(self, data: AnnData, train: bool) -> CombinedLoader:
        return CombinedLoader(
            get_dataloader_dict_from_anndata(data=data, cfg=self.cfg, train=train)
        )

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)


class BabelModel(ModelBase):
    def __init__(self, cfg):
        super(BabelModel, self).__init__()
        self.cfg = cfg
        self.model = BabelVAE(cfg)
        self.trainer = pl.Trainer(
            max_epochs=cfg.max_epochs,
            logger=pl.loggers.TensorBoardLogger(LOGS_PATH, name=cfg.model_name),
            callbacks=(
                [
                    pl.callbacks.EarlyStopping(
                        monitor="val_loss",
                        min_delta=cfg.min_delta,
                        patience=cfg.patience,
                        verbose=False,
                        mode="min",
                    )
                ]
                if cfg.early_stopping
                else []
            ),
        )

    def fit(self, train_anndata: AnnData, val_anndata: AnnData | None = None):
        train_loader = get_dataloader_dict_from_anndata(
            data=train_anndata, cfg=self.cfg, train=True
        )
        val_loader = (
            get_dataloader_dict_from_anndata(
                data=val_anndata, cfg=self.cfg, train=False
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
