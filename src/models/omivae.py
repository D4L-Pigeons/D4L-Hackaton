from argparse import Namespace
from typing import Tuple

import anndata as ad
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from anndata import AnnData
from torch import Tensor

from models.building_blocks import Block, ResidualBlock, ShortcutBlock
from models.ModelBase import ModelBase
from utils.paths import LOGS_PATH
from utils.loss_utils import (
    adt_reconstruction_loss,
    gex_reconstruction_loss,
    kld_stdgaussian,
)

from utils.data_utils import get_dataloader_from_anndata, get_dataset_from_anndata

class OmiVAEGaussian(pl.LightningModule):
    def __init__(self, cfg: Namespace):
        super(OmiVAEGaussian, self).__init__()
        self.assert_cfg(cfg)
        self.cfg = cfg
        self.fstmod_in = ShortcutBlock(
            input_size=cfg.first_modality_dim,
            output_size=cfg.first_modality_embedding_dim,
            hidden_size=cfg.first_modality_hidden_dim,
            batch_norm=cfg.batch_norm,
        )
        self.sndmod_in = ShortcutBlock(
            input_size=cfg.second_modality_dim,
            output_size=cfg.second_modality_embedding_dim,
            hidden_size=cfg.second_modality_hidden_dim,
            batch_norm=cfg.batch_norm,
        )
        self.encoder = nn.Sequential(
            Block(
                input_size=cfg.first_modality_embedding_dim
                + cfg.second_modality_embedding_dim,
                output_size=cfg.latent_dim * 2,
                hidden_size=cfg.hidden_dim,
                batch_norm=cfg.batch_norm,
            ),
            nn.SiLU(),
            ResidualBlock(
                input_size=cfg.latent_dim * 2,
                hidden_size=cfg.hidden_dim,
                batch_norm=cfg.batch_norm,
            ),
        )
        self.decoder = nn.Sequential(
            ShortcutBlock(
                input_size=cfg.latent_dim,
                output_size=cfg.latent_dim * 2,
                hidden_size=cfg.hidden_dim,
                batch_norm=cfg.batch_norm,
            ),
            nn.SiLU(),
            Block(
                input_size=cfg.latent_dim * 2,
                output_size=cfg.first_modality_embedding_dim
                + cfg.second_modality_embedding_dim,
                hidden_size=cfg.hidden_dim,
                batch_norm=cfg.batch_norm,
            ),
        )
        self.fstmod_out = ShortcutBlock(
            input_size=cfg.first_modality_embedding_dim,
            output_size=cfg.first_modality_dim,
            hidden_size=cfg.first_modality_hidden_dim,
            batch_norm=cfg.batch_norm,
        )
        self.sndmod_out = ShortcutBlock(
            input_size=cfg.second_modality_embedding_dim,
            output_size=cfg.second_modality_dim,
            hidden_size=cfg.second_modality_hidden_dim,
            batch_norm=cfg.batch_norm,
        )
        if cfg.classification_head:
            self.classification_head = Block(
                input_size=cfg.latent_dim,
                output_size=cfg.num_classes,
                hidden_size=cfg.num_classes * 2,
                batch_norm=cfg.batch_norm,
            )

    def _encode(self, x_fst: Tensor, x_snd: Tensor) -> Tensor:
        x_fst = self.fstmod_in(x_fst)
        x_snd = self.sndmod_in(x_snd)
        mu, logvar = self.encoder(torch.cat([x_fst, x_snd], dim=1)).chunk(2, dim=1)
        return mu, logvar

    def _decode(self, z: Tensor) -> Tuple[Tensor]:
        x_fst, x_snd = self.decoder(z).split(
            [
                self.cfg.first_modality_embedding_dim,
                self.cfg.second_modality_embedding_dim,
            ],
            dim=1,
        )
        x_fst = self.fstmod_out(x_fst)
        x_snd = self.sndmod_out(x_snd)
        return x_fst, x_snd

    def _get_decoder_jacobian(self, z: Tensor) -> Tensor:
        return torch.autograd.functional.jacobian(self.decoder, z)

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.size())
        return mu + eps * std

    def _forward_unsupervised(self, x_fst: Tensor, x_snd: Tensor) -> Tuple[Tensor]:
        mu, logvar = self._encode(x_fst, x_snd)
        z = self._reparameterize(mu, logvar)
        x_fst_hat, x_snd_hat = self._decode(z)
        return x_fst_hat, x_snd_hat, mu, logvar

    def _training_step_unsupervised(self, batch: Tensor) -> Tensor:
        x_fst, x_snd = batch
        x_fst_hat, x_snd_hat, mu, logvar = self._forward_unsupervised(x_fst, x_snd)
        recon_loss = F.mse_loss(x_fst_hat, x_fst) + F.mse_loss(x_snd_hat, x_snd)
        kld_loss = kld_stdgaussian(mu, logvar)

        return recon_loss, kld_loss

    def _forward_superivsed(self, x_fst: Tensor, x_snd: Tensor) -> Tuple[Tensor]:
        x_fst_hat, x_snd_hat, mu, logvar = self._forward_unsupervised(x_fst, x_snd)
        logits = self.classification_head(mu)  # classification performed with mu
        return x_fst_hat, x_snd_hat, logits, mu, logvar

    def _training_step_supervised(self, batch: Tensor) -> Tensor:
        x_fst, x_snd, target = batch
        x_fst_hat, x_snd_hat, logits, mu, logvar = self._forward_superivsed(
            x_fst, x_snd
        )
        recon_loss = gex_reconstruction_loss(
            x_fst_hat, x_fst
        ) + adt_reconstruction_loss(x_snd_hat, x_snd)
        kld_loss = kld_stdgaussian(mu, logvar)
        c_loss = F.cross_entropy(logits, target, weight=self.cfg.class_weights)

        return recon_loss, kld_loss, c_loss

    def training_step(self, batch: Tensor) -> Tensor:
        print("Is model supervised:?:", self.cfg.classification_head)
        if self.cfg.classification_head:
            recon_loss, kld_loss, c_loss = self._training_step_supervised(batch)
            self.log("Train recon", recon_loss, on_epoch=True, prog_bar=True)
            self.log("Train kld", kld_loss, on_epoch=True, prog_bar=True)
            self.log("Train class", c_loss, on_epoch=True, prog_bar=True)
            # print("Train recon", recon_loss)
            return (
                self.cfg.recon_loss_coef * recon_loss
                + self.cfg.kld_loss_coef * kld_loss
                + self.cfg.c_loss_coef * c_loss
            )
        else:
            recon_loss, kld_loss = self._training_step_unsupervised(batch)
            self.log("Train recon", recon_loss, on_epoch=True, prog_bar=True)
            self.log("Train kld", kld_loss, on_epoch=True, prog_bar=True)
            # print("Train recon", recon_loss)
            return (
                self.cfg.recon_loss_coef * recon_loss
                + self.cfg.kld_loss_coef * kld_loss
            )

    def validation_step(self, batch: Tensor) -> Tensor:
        if self.cfg.classification_head:
            recon_loss, kld_loss, c_loss = self._training_step_supervised(batch)
            self.log("Val recon", recon_loss, on_epoch=True, prog_bar=True)
            self.log("Val kld", kld_loss, on_epoch=True, prog_bar=True)
            self.log("Val class", c_loss, on_epoch=True, prog_bar=True)
        else:
            recon_loss, kld_loss = self._training_step_unsupervised(batch)
            self.log("Val recon", recon_loss, on_epoch=True, prog_bar=True)
            self.log("Val kld", kld_loss, on_epoch=True, prog_bar=True)

    def predict(self, x: Tensor) -> Tensor:
        if not self.classification_head:
            raise ValueError("Model does not have a classification head")
        mu, _ = self._encode(x)
        logits = self.classification_head(mu)
        return logits

    def predict_proba(self, data: AnnData):
        if not self.classification_head:
            raise ValueError("Model does not have a classification head")
        mu, _ = self._encode(data)
        logits = self.classification_head(mu)
        return torch.softmax(logits, dim=1)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)

    def assert_cfg(self, cfg: Namespace) -> None:
        assert hasattr(cfg, "max_epochs"), AttributeError(
            'cfg does not have the attribute "max_epochs"'
        )
        assert hasattr(cfg, "log_every_n_steps"), AttributeError(
            'cfg does not have the attribute "log_every_n_steps"'
        )
        # assert hasattr(cfg, "logger"), AttributeError(
        #     'cfg does not have the attribute "logger"'
        # )
        assert hasattr(cfg, "classification_head"), AttributeError(
            'cfg does not have the attribute "classification_head"'
        )
        assert hasattr(cfg, "first_modality_hidden_dim"), AttributeError(
            'cfg does not have the attribute "first_modality_hidden_dim"'
        )
        assert hasattr(cfg, "second_modality_hidden_dim"), AttributeError(
            'cfg does not have the attribute "second_modality_hidden_dim"'
        )
        assert hasattr(cfg, "first_modality_embedding_dim"), AttributeError(
            'cfg does not have the attribute "first_modality_embedding_dim"'
        )
        assert hasattr(cfg, "second_modality_embedding_dim"), AttributeError(
            'cfg does not have the attribute "second_modality_embedding_dim"'
        )
        assert hasattr(cfg, "recon_loss_coef"), AttributeError(
            'cfg does not have the attribute "recon_loss_coef"'
        )
        assert hasattr(cfg, "kld_loss_coef"), AttributeError(
            'cfg does not have the attribute "kld_loss_coef"'
        )
        if cfg.classification_head:
            assert hasattr(cfg, "num_classes"), AttributeError(
                'cfg does not have the attribute "num_classes"'
            )
            assert hasattr(cfg, "class_weights"), AttributeError(
                'cfg does not have the attribute "class_weights"'
            )
            assert hasattr(cfg, "c_loss_coef"), AttributeError(
                'cfg does not have the attribute "c_loss_coef"'
            )
        assert hasattr(cfg, "lr"), AttributeError(
            'cfg does not have the attribute "lr"'
        )


class OmiIWAE(OmiVAEGaussian):
    def __init__(self, cfg: Namespace):
        super(OmiIWAE, self).__init__(cfg)
        self.num_samples = cfg.num_samples

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn((self.num_samples, *std.size()), device=std.device)
        z = mu.unsqueeze(0) + eps * std.unsqueeze(0)
        return z

    def _forward_unsupervised(self, x_fst: Tensor, x_snd: Tensor) -> Tuple[Tensor]:
        mu, logvar = self._encode(x_fst, x_snd)
        z = self._reparameterize(mu, logvar)
        x_fst_hat, x_snd_hat = self._decode(z)
        return x_fst_hat, x_snd_hat, mu, logvar

    def _training_step_unsupervised(self, batch: Tensor) -> Tensor:
        x_fst, x_snd = batch
        x_fst_hat, x_snd_hat, mu, logvar = self._forward_unsupervised(x_fst, x_snd)
        recon_loss = F.mse_loss(
            x_fst_hat, x_fst.unsqueeze(0).expand(self.num_samples, *x_fst.size())
        ) + F.mse_loss(
            x_snd_hat, x_snd.unsqueeze(0).expand(self.num_samples, *x_snd.size())
        )
        kld_loss = kld_stdgaussian(mu, logvar)

        return recon_loss, kld_loss

    def _forward_supervised(self, x_fst: Tensor, x_snd: Tensor) -> Tuple[Tensor]:
        x_fst_hat, x_snd_hat, mu, logvar = self._forward_unsupervised(x_fst, x_snd)
        logits = self.classification_head(
            mu.mean(dim=0)
        )  # classification performed with average mu
        return x_fst_hat, x_snd_hat, logits, mu, logvar

    def _training_step_supervised(self, batch: Tensor) -> Tensor:
        x_fst, x_snd, target = batch
        x_fst_hat, x_snd_hat, logits, mu, logvar = self._forward_supervised(
            x_fst, x_snd
        )
        recon_loss = gex_reconstruction_loss(
            x_fst_hat, x_fst.unsqueeze(0).expand(self.num_samples, *x_fst.size())
        ) + adt_reconstruction_loss(
            x_snd_hat, x_snd.unsqueeze(0).expand(self.num_samples, *x_snd.size())
        )
        kld_loss = kld_stdgaussian(mu, logvar)
        c_loss = F.cross_entropy(logits, target, weight=self.cfg.class_weights)

        return recon_loss, kld_loss, c_loss

    def assert_cfg(self, cfg: Namespace) -> None:
        assert hasattr(cfg, "num_samples"), AttributeError(
            'cfg does not have the attribute "num_samples"'
        )


_OMIVAE_IMPLEMENTATIONS = {
    "OmiVAEGaussian": OmiVAEGaussian,
    "OmiIWAE": OmiIWAE,
}


class OmiModel(ModelBase):
    def __init__(self, cfg: Namespace):
        super(OmiModel, self).__init__()
        self.assert_cfg(cfg)
        self.cfg = cfg
        self.model = _OMIVAE_IMPLEMENTATIONS[cfg.omivae_implementation](cfg)
        self.trainer = pl.Trainer(
            max_epochs=cfg.max_epochs,
            log_every_n_steps=cfg.log_every_n_steps,
            logger=pl.loggers.TensorBoardLogger(
                LOGS_PATH, name=cfg.omivae_implementation
            ),
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
                if self.cfg.early_stopping
                else None
            ),
        )

    def train(self, train_data: AnnData, val_data: AnnData = None) -> None:
        self.trainer.fit(
            model=self.model,
            train_dataloaders=get_dataloader_from_anndata(
                train_data,
                self.cfg.first_modality_dim,
                self.cfg.second_modality_dim,
                self.cfg.batch_size,
                shuffle=True,
                include_class_labels=self.cfg.classification_head,
            ),
            val_dataloaders=(
                get_dataset_from_anndata(
                    val_data,
                    self.cfg.first_modality_dim,
                    self.cfg.second_modality_dim,
                    include_class_labels=self.cfg.classification_head,
                )
                if val_data is not None
                else None
            ),
        )

    def predict(self, data: AnnData):
        return self.model.predict(data)

    def predict_proba(self, data: AnnData):
        return self.model.predict_proba(data)

    def save(self, file_path: str):
        save_path = file_path + ".ckpt"
        torch.save(self.model.state_dict(), save_path)
        return save_path

    def load(self, file_path: str):
        load_path = file_path + ".ckpt"
        self.model.load_state_dict(torch.load(load_path))

    def assert_cfg(self, cfg: Namespace) -> None:
        self.assert_cfg_general(cfg)
        assert hasattr(cfg, "omivae_implementation"), AttributeError(
            'cfg does not have the attribute "omivae_implementation"'
        )
        assert hasattr(cfg, "output_modelling_type"), AttributeError(
            'cfg does not have the attribute "output_modelling_type"'
        )
        assert cfg.output_modelling_type in [
            "mse_direct_reconstruction",
            "ll_neg_binomial",
        ], ValueError(
            f"Invalid output modelling type: {cfg.output_modelling_type}. Must be one of ['mse_direct_reconstruction', 'll_neg_binomial']"
        )
        assert cfg.omivae_implementation in _OMIVAE_IMPLEMENTATIONS, ValueError(
            f"Invalid OmiVAE implementation: {cfg.omivae_implementation}"
        )
        assert hasattr(cfg, "early_stopping"), AttributeError(
            'cfg does not have the attribute "early_stopping"'
        )
        if cfg.early_stopping:
            assert hasattr(cfg, "min_delta"), AttributeError(
                'cfg does not have the attribute "min_delta"'
            )
            assert hasattr(cfg, "patience"), AttributeError(
                'cfg does not have the attribute "patience"'
            )


# class OmiVAE(OmiAE):
