from argparse import Namespace
from typing import Dict, Optional, Tuple, Union

import anndata as ad
import pytorch_lightning as pl
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from anndata import AnnData
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from sklearn.metrics import balanced_accuracy_score
from torch import Tensor
from torch.utils.data import DataLoader

from models.ModelBase import ModelBase
from utils.data_utils import (
    get_dataloader_dict_from_anndata,
    get_dataloader_from_anndata,
)
from utils.paths import LOGS_PATH


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

    def predict(self, x):
        encoded_dict = {}
        for modality_name, modality_data in x.items():
            encoded_dict[modality_name] = self.encode(modality_data)[1]  # mu
        return encoded_dict


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

    def encode_anndata(self, anndata):
        adt_mask = anndata.var["feature_types"] == "ADT"
        
        adt_anndata = anndata[:, adt_mask]
        gex_anndata = anndata[:, ~adt_mask]
        
        adt_latent = []
        data = torch.tensor(adt_anndata.X.A)
        self.model["adt"].eval() # we need to set the model to evaluation mode, so that the dropout is no longer considered
        z = self.model["adt"].encoder(data)
        adt_latent += [z]
        adt_latent = torch.cat(adt_latent).detach().cpu().numpy()
        
        anndata.obsm["latent_embedding_adt"] = adt_latent
        
        gex_latent = []
        data = torch.tensor(gex_anndata.X.A)
        self.model["gex"].eval() # we need to set the model to evaluation mode, so that the dropout is no longer considered
        z = self.model["gex"].encoder(data)
        gex_latent += [z]
        gex_latent = torch.cat(gex_latent).detach().cpu().numpy()
        
        anndata.obsm["latent_embedding_gex"] = gex_latent
    

    def training_step(
        self, batch: Tuple[Tensor], batch_idx: int
    ) -> Tuple[Tensor, Dict[str, float]]:
        total_loss = 0.0
        full_losses_dict = {}
        # for encoding_modality_name, encoding_model in self.model.items():
        #     (x_enc,) = batch[encoding_modality_name]
        #     encoded, mu, std = encoding_model.encode(x_enc)
        #     kld_loss = self.kld_divergence(mu, std)
        #     full_losses_dict[f"{encoding_modality_name}_kld"] = kld_loss.detach().item()
        #     total_loss += self.cfg.kld_loss_coef * kld_loss

        #     for decoding_modality_name, decoding_model in self.model.items():
        #         (x_dec,) = batch[decoding_modality_name]
        #         decoded = decoding_model.decode(encoded)
        #         recon_loss = F.mse_loss(decoded, x_dec)
        #         total_loss += self.cfg.recon_loss_coef * recon_loss
        #         full_losses_dict[
        #             f"{encoding_modality_name}_{decoding_modality_name}_recon"
        #         ] = recon_loss.detach().item()
        for x_enc, (encoding_modality_name, encoding_model) in zip(
            batch, self.model.items()
        ):
            encoded, mu, std = encoding_model.encode(x_enc)
            kld_loss = self.kld_divergence(mu, std)
            full_losses_dict[f"{encoding_modality_name}_kld"] = kld_loss.detach().item()
            total_loss += self.cfg.kld_loss_coef * kld_loss
            for x_dec, (decoding_modality_name, decoding_model) in zip(
                batch, self.model.items()
            ):
                decoded = decoding_model.decode(encoded)
                recon_loss = F.mse_loss(decoded, x_dec)
                total_loss += self.cfg.recon_loss_coef * recon_loss
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
        # train_loader = get_dataloader_dict_from_anndata(
        #     data=train_anndata, cfg=self.cfg, train=True
        # )
        train_loader = get_dataloader_from_anndata(
            data=train_anndata,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            include_class_labels=False,
        )
        val_loader = (
            get_dataloader_from_anndata(
                data=val_anndata,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                include_class_labels=False,
            )
            if val_anndata is not None
            else None
        )

        self.trainer.fit(
            model=self.model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

    def predict(self, data: AnnData) -> Dict[str, Tensor]:
        print("predict in omivae module")
        latent_representation_dict = self.trainer.predict(
            model=self.model,
            dataloaders=get_dataloader_from_anndata(
                data,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                first_modality_dim=self.cfg.first_modality_dim,
                second_modality_dim=self.cfg.second_modality_dim,
                include_class_labels=self.cfg.classification_head
                or self.cfg.include_class_labels,
                target_hierarchy_level=self.cfg.target_hierarchy_level,
            ),
        )
        for modality_name, latent_representation in latent_representation_dict.items():
            latent_representation_dict[modality_name] = torch.cat(
                latent_representation, dim=0
            )
        return latent_representation_dict

    def save(self, file_path: str):
        save_path = file_path + ".ckpt"
        torch.save(self.model.state_dict(), save_path)
        return save_path

    def load(self, file_path: str):
        load_path = file_path + ".ckpt"
        self.model.load_state_dict(torch.load(load_path))

    def assert_cfg(self, cfg: Namespace) -> None:
        pass
