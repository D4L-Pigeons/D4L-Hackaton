from models.ModelBase import ModelBaseInterface
from models.building_blocks import Block, ResidualBlock, ShortcutBlock
from utils.loss_utils import (
    kld_stdgaussian,
    gex_reconstruction_loss,
    adt_reconstruction_loss,
)
from utils.data_utils import get_dataset_from_anndata, get_dataloader_from_anndata
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from argparse import Namespace
import anndata as ad
from anndata import AnnData
from typing import Tuple
from torch import Tensor


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
        eps = torch.randn(self.cfg.batch_size, self.cfg.latent_dim)
        return mu + eps * std

    def _forward_usupervised(self, x_fst: Tensor, x_snd: Tensor) -> Tuple[Tensor]:
        mu, logvar = self._encode(x_fst, x_snd)
        z = self._reparameterize(mu, logvar)
        x_fst_hat, x_snd_hat = self._decode(z)
        return x_fst_hat, x_snd_hat, mu, logvar

    def _training_step_unsupervised(self, batch: Tensor) -> Tensor:
        x_fst, x_snd = batch
        x_fst_hat, x_snd_hat, mu, logvar = self._forward_usupervised(x_fst, x_snd)
        recon_loss = F.mse_loss(x_fst_hat, x_fst) + F.mse_loss(x_snd_hat, x_snd)
        kld_loss = kld_stdgaussian(mu, logvar)

        self.log("Reconstruction Loss", recon_loss)
        self.log("KLD Loss", kld_loss)

        return self.cfg.recon_loss_coef * recon_loss + self.cfg.kld_loss_coef * kld_loss

    def _forward_superivsed(self, x_fst: Tensor, x_snd: Tensor) -> Tuple[Tensor]:
        x_fst_hat, x_snd_hat, mu, logvar = self._forward_usupervised(x_fst, x_snd)
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
        self.log("Reconstruction Loss", recon_loss)
        self.log("KLD Loss", kld_loss)
        self.log("Classification Loss", c_loss)

        return (
            self.cfg.recon_loss_coef * recon_loss
            + self.cfg.kld_loss_coef * kld_loss
            + self.cfg.c_loss_coef * c_loss
        )

    def training_step(self, batch: Tensor) -> Tensor:
        if self.cfg.classification_head:
            return self._training_step_supervised(batch)
        return self._training_step_unsupervised(batch)

    def validation_step(self, batch: Tensor) -> Tensor:
        if self.cfg.classification_head:
            return self._training_step_supervised(batch)

    def _predict(self, x: Tensor) -> Tensor:
        if not self.classification_head:
            raise ValueError("Model does not have a classification head")
        mu, _ = self._encode(x)
        logits = self.classification_head(mu)
        return logits

    def _predict_proba(self, data: AnnData):
        if not self.classification_head:
            raise ValueError("Model does not have a classification head")
        mu, _ = self._encode(data)
        logits = self.classification_head(mu)
        return torch.softmax(logits, dim=1)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)

    def assert_cfg(self, cfg: Namespace) -> None:
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
        assert hasattr(cfg, "class_weights"), AttributeError(
            'cfg does not have the attribute "class_weights"'
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
            assert hasattr(cfg, "c_loss_coef"), AttributeError(
                'cfg does not have the attribute "c_loss_coef"'
            )
        assert hasattr(cfg, "lr"), AttributeError(
            'cfg does not have the attribute "lr"'
        )


__OMIVAE_IMPLEMENTATIONS = {
    "OmiVAEGaussian": OmiVAEGaussian,
}


class OmiInterface(ModelBaseInterface):
    def __init__(self, cfg: Namespace):
        super(OmiInterface, self).__init__()
        self.assert_cfg(cfg)
        self.cfg = cfg
        self.model = __OMIVAE_IMPLEMENTATIONS[cfg.omivae_implementation](cfg)
        self.trainer = pl.Trainer(
            max_epochs=cfg.max_epochs,
            progress_bar_refresh_rate=cfg.progress_bar_refresh_rate,
            logger=cfg.logger,
        )

    def train(self, train_data: AnnData, val_data: AnnData = None) -> None:
        self.trainer.fit(
            model=self.model,
            train_data=get_dataloader_from_anndata(
                train_data,
                self.cfg.batch_size,
                shuffle=True,
                include_class_labels=self.cfg.classification_head,
            ),
            val_data=(
                get_dataset_from_anndata(
                    val_data,
                    include_class_labels=self.cfg.classification_head,
                )
                if val_data is not None
                else None
            ),
        )

    def predict(self, data: AnnData):
        pass

    def predict_proba(self, data: AnnData):
        pass

    def save(self, file_path: str):
        pass

    def load(self, file_path: str):
        pass

    def assert_cfg(self, cfg: Namespace) -> None:
        self.assert_cfg_general(cfg)
        assert hasattr(cfg, "omivae_implementation"), AttributeError(
            'cfg does not have the attribute "omivae_implementation"'
        )
        assert hasattr(cfg, "max_epochs"), AttributeError(
            'cfg does not have the attribute "max_epochs"'
        )
        assert hasattr(cfg, "progress_bar_refresh_rate"), AttributeError(
            'cfg does not have the attribute "progress_bar_refresh_rate"'
        )
        assert hasattr(cfg, "logger"), AttributeError(
            'cfg does not have the attribute "logger"'
        )


# class OmiVAE(OmiAE):

# class OmiIWAE(OmiAE):
