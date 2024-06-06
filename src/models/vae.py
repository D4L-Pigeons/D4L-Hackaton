
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
import pytorch_lightning as pl
from argparse import Namespace

from models.building_blocks import Block, ShortcutBlock


class VAE(pl.LightningModule):
    def __init__(self, cfg: Namespace):
        super(VAE, self).__init__()
        self.assert_cfg(cfg)
        self.cfg = cfg
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
                        mode="min"
                    )
                ]
                if cfg.early_stopping
                else []
            )
        )

        self.mod_in = ShortcutBlock(
            input_size=cfg.modality_dim,
            output_size=cfg.modality_embedding_dim,
            hidden_size=cfg.modality_hidden_dim,
            batch_norm=cfg.batch_norm,
        )

        self.mu = Block(
            input_size=cfg.modality_embedding_dim,
            output_size=cfg.latent_dim,
            hidden_size=cfg.latent_hidden_dim,
            batch_norm=cfg.batch_norm,
        )

        self.logvar = Block(
            input_size=cfg.modality_embedding_dim,
            output_size=cfg.latent_dim,
            hidden_size=cfg.latent_hidden_dim,


            batch_norm=cfg.batch_norm,
        )

        self.decoder = nn.Sequential(
            Block(
                input_size=cfg.latent_dim,
                output_size=cfg.modality_embedding_dim,
                hidden_size=cfg.latent_hidden_dim,
                batch_norm=cfg.batch_norm,
            ),
            ShortcutBlock(
                input_size=cfg.modality_embedding_dim,
                output_size=cfg.modality_dim,
                hidden_size=cfg.modality_hidden_dim,
                batch_norm=cfg.batch_norm,
            ),
        )

        self.kld_weight = cfg.kld_weight
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.mod_in(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kld(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def get_modality(self, batch):
        if self.cfg.modality == "GEX":
            return batch[0]
        elif self.cfg.modality == "ADT":
            return batch[1]
    def training_step(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        x, _ = self.get_modality(batch)
        x_hat, mu, logvar = self(x)
        reconstruction_loss = self.loss(x_hat, x)
        kld_loss = self.kld_weight * self.kld(mu, logvar)
        loss = reconstruction_loss + kld_loss

        self.log_dict(
            {
                "reconstruction_loss": reconstruction_loss,
                "kld_loss": kld_loss,
                "loss": loss,
            }
        )

        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        x, _ = batch
        x_hat, mu, logvar = self(x)
        reconstruction_loss = self.loss(x_hat, x)
        kld_loss = self.kld_weight * self.kld(mu, logvar)
        loss = reconstruction_loss + kld_loss

        self.log_dict(
            {
                "val_reconstruction_loss": reconstruction_loss,
                "val_kld_loss": kld_loss,
                "val_loss": loss,
            }
        )

        return loss

    def train(self, train_data: AnnData, val_data: AnnData = None) -> None:
        self.trainer.fit(
            model=self.model,
            train_dataloaders=get_dataloader_from_anndata(
                train_data,
                self.cfg.first_modality_dim,
                self.cfg.second_modality_dim,
                self.cfg.batch_size,
                shuffle=True,

            ),
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)

    def assert_cfg(self, cfg: Namespace) -> None:
        assert hasattr(cfg, "modality"), AttributeError(
            'cfg does not have the attribute "modality"'
        )
        assert hasattr(cfg, "modality_dim"), AttributeError(
            'cfg does not have the attribute "modality_dim"'
        )
        assert hasattr(cfg, "modality_embedding_dim"), AttributeError(
            'cfg does not have the attribute "modality_embedding_dim"'
        )
        assert hasattr(cfg, "modality_hidden_dim"), AttributeError(
            'cfg does not have the attribute "modality_hidden_dim"'
        )
        assert hasattr(cfg, "latent_dim"), AttributeError(
            'cfg does not have the attribute "latent_dim"'
        )
        assert hasattr(cfg, "latent_hidden_dim"), AttributeError(
            'cfg does not have the attribute "latent_hidden_dim"'
        )
        assert hasattr(cfg, "kld_weight"), AttributeError(
            'cfg does not have the attribute "kld_weight"'
        )
        assert hasattr(cfg, "learning_rate"), AttributeError(
            'cfg does not have the attribute "learning_rate"'
        )
        assert hasattr(cfg, "batch_norm"), AttributeError(
            'cfg does not have the attribute "batch_norm"'
        )