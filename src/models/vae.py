from argparse import Namespace
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from anndata import AnnData
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from tqdm import tqdm

from models.building_blocks import Block, ShortcutBlock
from models.ModelBase import ModelBase
from utils.data_utils import get_dataloader_dict_from_anndata
from utils.paths import LOGS_PATH


class SingleModalityVAE(nn.Module):
    def __init__(self, cfg_name: str, modality_cfg: Namespace):
        super(SingleModalityVAE, self).__init__()
        self.assert_modality_cfg(cfg_name, modality_cfg)
        self.cfg_name = cfg_name
        self.modality_cfg = modality_cfg

        self.mod_in = ShortcutBlock(
            input_size=modality_cfg.dim,
            output_size=modality_cfg.embedding_dim,
            hidden_size=modality_cfg.hidden_dim,
            batch_norm=modality_cfg.batch_norm,
        )

        self.mu = Block(
            input_size=modality_cfg.embedding_dim,
            output_size=modality_cfg.latent_dim,
            hidden_size=modality_cfg.latent_hidden_dim,
            batch_norm=modality_cfg.batch_norm,
        )

        self.logvar = Block(
            input_size=modality_cfg.embedding_dim,
            output_size=modality_cfg.latent_dim,
            hidden_size=modality_cfg.latent_hidden_dim,
            batch_norm=modality_cfg.batch_norm,
        )

        self.decoder = nn.Sequential(
            Block(
                input_size=modality_cfg.latent_dim,
                output_size=modality_cfg.embedding_dim,
                hidden_size=modality_cfg.latent_hidden_dim,
                batch_norm=modality_cfg.batch_norm,
            ),
            ShortcutBlock(
                input_size=modality_cfg.embedding_dim,
                output_size=modality_cfg.dim,
                hidden_size=modality_cfg.hidden_dim,
                batch_norm=modality_cfg.batch_norm,
            ),
        )

        self.kld_weight = modality_cfg.kld_weight
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

    def training_step(self, batch: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        (x,) = batch
        x_hat, mu, logvar = self(x)
        reconstruction_loss = self.loss(x_hat, x)
        kld_loss = self.kld_weight * self.kld(mu, logvar)
        loss = reconstruction_loss + kld_loss

        losses_dict = {
            f"{self.cfg_name}_loss": loss,
            f"{self.cfg_name}_reconstruction_loss": reconstruction_loss,
            f"{self.cfg_name}_kld_loss": kld_loss,
        }
        return loss, losses_dict

    def validation_step(self, batch: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        (x,) = batch
        x_hat, mu, logvar = self(x)
        reconstruction_loss = self.loss(x_hat, x)
        kld_loss = self.kld_weight * self.kld(mu, logvar)
        loss = reconstruction_loss + kld_loss

        losses_dict = {
            f"val_{self.cfg_name}_loss": loss,
            f"val_{self.cfg_name}_reconstruction_loss": reconstruction_loss,
            f"val_{self.cfg_name}_kld_loss": kld_loss,
        }
        return loss, losses_dict

    def predict(self, batch: Tensor) -> Tensor:
        (x,) = batch
        x = self.mod_in(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.reparameterize(mu, logvar)
        return z


class VAE(pl.LightningModule, ModelBase):
    def __init__(self, cfg: Namespace):
        super(VAE, self).__init__()
        self.assert_cfg(cfg)
        self.cfg = cfg
        self.automatic_optimization = False
        self.model = nn.ModuleDict(
            {
                cfg_name: SingleModalityVAE(cfg_name, modality_cfg)
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

    def combine_steps(
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        step_outputs = [
            model.training_step(batch[cfg_name])
            for cfg_name, model in self.model.items()
        ]
        losses, losses_dicts = zip(*step_outputs)
        losses_dicts = {k: v for d in losses_dicts for k, v in d.items()}
        loss = sum(losses)
        return loss, losses_dicts

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> STEP_OUTPUT:
        loss, losses_dicts = self.combine_steps(batch, batch_idx)
        loss.backward()
        for optimizer in self.optimizers():
            optimizer.step()
        self.log_dict(
            losses_dicts,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, *args: Any, **kwargs: Any) -> Tensor:
        loss, losses_dicts = self.combine_steps(*args, **kwargs)
        self.log_dict(
            losses_dicts,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def get_dataloader(self, data: AnnData, train: bool) -> CombinedLoader:
        return CombinedLoader(
            get_dataloader_dict_from_anndata(
                data,
                self.cfg,
                train=train,
            ),
            mode="max_size",  # if train else "sequential",
        )

    def fit(self, train_data: AnnData, val_data: AnnData = None) -> None:
        self.trainer.fit(
            model=self,
            train_dataloaders=self.get_dataloader(train_data, train=True),
            val_dataloaders=(
                self.get_dataloader(val_data, train=False)
                if val_data is not None
                else None
            ),
        )

    def predict_batch(self, batch):
        predict_results = {
            cfg_name: model.predict(batch[cfg_name])
            for cfg_name, model in self.model.items()
        }
        return torch.dstack(list(predict_results.values()))

    def predict(self, data: AnnData) -> Tensor:
        self.eval()
        results = []
        with torch.no_grad():
            dataloader = self.get_dataloader(data, train=False)
            for batch in tqdm(iter(dataloader)):
                results.append(self.predict_batch(batch))
        return torch.vstack(results)

    def save(self, file_path: str) -> str:
        save_path = file_path + ".ckpt"
        torch.save(self.model.state_dict(), save_path)
        return save_path

    def load(self, file_path: str) -> None:
        self.load_state_dict(torch.load(file_path))

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(model.parameters(), lr=model.modality_cfg.lr)
            for model in self.model.values()
        ]
        return optimizers
