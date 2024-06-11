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


class OmiAE(pl.LightningModule):
    def __init__(self, cfg: Namespace):
        super(OmiAE, self).__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.save_hyperparameters()

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def training_step(self, batch, batch_idx):
        x1, x2 = batch
        x = torch.cat((x1, x2), dim=-1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("Train loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.lr)
        return optimizer


class OmiGMPriorProbabilisticAE(pl.LightningModule):
    def __init__(self, cfg: Namespace):
        super(OmiGMPriorProbabilisticAE, self).__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.save_hyperparameters()
        # learnable logits of GMM components
        self.component_logits = nn.Parameter(
            data=torch.zeros(size=(cfg.no_components,)), requires_grad=True
        )
        # learnable means of GMM
        self.means = nn.Parameter(
            torch.randn(cfg.no_components, cfg.latent_dim), requires_grad=True
        )
        # fixed STDs of GMM
        self.register_buffer(
            "stds", cfg.components_std * torch.ones(cfg.no_components, cfg.latent_dim)
        )

    def training_step(self, batch: Tuple[Tensor]):
        x1, x2, labels = batch
        batch_size = x1.shape[0]
        x = torch.cat((x1, x2), dim=-1)
        z_means, z_stds = self.encoder(x).chunk(2, dim=-1)
        z_stds = self._var_transformation(z_stds)
        normal_rv = self._make_normal_rv(z_means, z_stds)
        entropy_per_batch_sample = (
            normal_rv.entropy().sum(dim=-1).unsqueeze(0)
        )  # (1, batch_size)
        assert entropy_per_batch_sample.shape == (
            1,
            batch_size,
        ), AssertionError(
            f"Entropy shape is {entropy_per_batch_sample.shape}, expected {(1, batch_size)}"
        )
        z_sample = normal_rv.rsample(
            sample_shape=(self.cfg.no_latent_samples,)
        ).unsqueeze(
            2
        )  # (no_laten_samples, batch_size, 1, latent_dim)
        assert z_sample.shape == (
            self.cfg.no_latent_samples,
            batch_size,
            1,
            self.cfg.latent_dim,
        ), AssertionError(
            f"z_sample shape is {z_sample.shape}, expected {(self.cfg.no_latent_samples, batch_size, 1, self.cfg.latent_dim)}"
        )
        gmm = self._make_gmm()
        per_component_log_prob = (
            -gmm.component_distribution.log_prob(  # log_prob into negative log_prob
                z_sample
            )
        )  # (no_laten_samples, batch_size, no_components)
        assert per_component_log_prob.shape == (
            self.cfg.no_latent_samples,
            batch_size,
            self.cfg.no_components,
        ), AssertionError(
            f"per_component_logprob shape is {per_component_log_prob.shape}, expected {(self.cfg.no_latent_samples, batch_size, self.cfg.no_components)}"
        )
        component_indicator = torch.arange(self.cfg.no_components).unsqueeze(0).repeat(
            (batch_size, 1)
        ) == labels.unsqueeze(1)
        assert component_indicator.shape == (
            batch_size,
            self.cfg.no_components,
        ), AssertionError(
            f"component_indicator shape is {component_indicator.shape}, expected {(batch_size, self.cfg.no_components)}"
        )
        gmm_likelihood_per_k = per_component_log_prob[
            :, component_indicator
        ]  # (no_laten_samples, batch_size)
        assert gmm_likelihood_per_k.shape == (
            self.cfg.no_latent_samples,
            batch_size,
        ), AssertionError(
            f"gmm_likelihood_per_k shape is {gmm_likelihood_per_k.shape}, expected {(self.cfg.no_latent_samples, batch_size)}"
        )
        x_hat = self.decoder(z_sample.squeeze(2))
        recon_loss_per_k = F.mse_loss(
            x_hat, x.repeat(self.cfg.no_latent_samples, 1, 1), reduction="none"
        ).mean(
            dim=-1
        )  # (no_laten_samples, batch_size)
        assert recon_loss_per_k.shape == (
            self.cfg.no_latent_samples,
            batch_size,
        ), AssertionError(
            f"recon_loss_per_k shape is {recon_loss_per_k.shape}, expected {(self.cfg.no_latent_samples, batch_size)}"
        )
        if self.cfg.no_latent_samples > 1:  # IWAE with multiple samples
            total_loss = torch.logsumexp(
                self.cfg.gmm_likelihood_loss_coef * gmm_likelihood_per_k
                + self.cfg.entropy_loss_coef * entropy_per_batch_sample
                + self.cfg.recon_loss_coef * recon_loss_per_k,
                dim=0,
            ).mean()
        else:  # IWAE reduces to VAE with single sample
            total_loss = (
                self.cfg.gmm_likelihood_loss_coef * gmm_likelihood_per_k
                + self.cfg.entropy_loss_coef * entropy_per_batch_sample
                + self.cfg.recon_loss_coef * recon_loss_per_k
            ).mean()
        self.log("Train loss", total_loss, on_epoch=True, prog_bar=True)
        self.log(
            "Reconstruction loss", recon_loss_per_k.mean(), on_epoch=True, prog_bar=True
        )
        self.log(
            "GMM likelihood loss",
            gmm_likelihood_per_k.mean(),
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "Entropy loss",
            entropy_per_batch_sample.mean(),
            on_epoch=True,
            prog_bar=True,
        )

        return total_loss

    def _var_transformation(self, std: Tensor) -> Tensor:
        return F.softplus(std) + 1e-6

    def _make_normal_rv(self, mu: Tensor, var: Tensor):
        return td.Normal(mu, var)

    def _make_gmm(self):
        categorical = td.Categorical(logits=self.component_logits)
        comp = td.Independent(
            td.Normal(self.means, self.stds), reinterpreted_batch_ndims=1
        )
        return td.MixtureSameFamily(categorical, comp)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.lr)
        return optimizer


_OMIIVAE_IMPLEMENTATIONS = {
    "OmiAE": OmiAE,
    "OmiGMPriorProbabilisticAE": OmiGMPriorProbabilisticAE,
}


class OmiModel(ModelBase):
    def __init__(self, cfg):
        super(OmiModel, self).__init__()
        self.cfg = cfg
        self.model = _OMIIVAE_IMPLEMENTATIONS[cfg.omivae_implementation](cfg)
        self.trainer = pl.Trainer(
            max_epochs=cfg.max_epochs,
            logger=pl.loggers.TensorBoardLogger(LOGS_PATH, name=cfg.model_name),
        )

    def fit(self, train_anndata: AnnData, val_anndata: AnnData | None = None):
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
