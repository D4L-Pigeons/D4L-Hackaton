from abc import ABC, abstractmethod
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
from sklearn.metrics import balanced_accuracy_score
from torch import Tensor

from models.building_blocks import Block, ResidualBlock, ShortcutBlock
from models.ModelBase import ModelBase
from utils.data_utils import get_dataloader_from_anndata, get_dataset_from_anndata
from utils.loss_utils import (
    adt_reconstruction_loss,
    gex_reconstruction_loss,
    kld_stdgaussian,
)
from utils.paths import LOGS_PATH


class OmiAE(pl.LightningModule):
    def __init__(self, cfg: Namespace):
        super(OmiAE, self).__init__()
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
                output_size=cfg.encoder_out_dim,
                hidden_size=cfg.encoder_hidden_dim,
                batch_norm=cfg.batch_norm,
            ),
            nn.SiLU(),
            ResidualBlock(
                input_size=cfg.encoder_out_dim,
                hidden_size=cfg.encoder_hidden_dim,
                batch_norm=cfg.batch_norm,
            ),
        )
        self.decoder = nn.Sequential(
            ShortcutBlock(
                input_size=cfg.decoder_in_dim,
                output_size=cfg.decoder_in_dim * 2,
                hidden_size=cfg.decoder_hidden_dim,
                batch_norm=cfg.batch_norm,
            ),
            nn.SiLU(),
            Block(
                input_size=cfg.decoder_in_dim * 2,
                output_size=cfg.first_modality_embedding_dim
                + cfg.second_modality_embedding_dim,
                hidden_size=cfg.decoder_hidden_dim,
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
                input_size=cfg.decoder_in_dim,
                output_size=cfg.num_classes,
                hidden_size=cfg.num_classes * 2,
                batch_norm=cfg.batch_norm,
            )

    def _encode(self, x_fst: Tensor, x_snd: Tensor) -> Tensor:
        x_fst = self.fstmod_in(x_fst)
        # print("encode 0 passed")
        x_snd = self.sndmod_in(x_snd)
        # print("encode 1 passed")
        encoder_out = self.encoder(torch.cat([x_fst, x_snd], dim=-1))

        return encoder_out

    def _decode(self, z: Tensor) -> Tuple[Tensor]:
        x_fst, x_snd = self.decoder(z).split(
            [
                self.cfg.first_modality_embedding_dim,
                self.cfg.second_modality_embedding_dim,
            ],
            dim=-1,
        )
        # print("decode 0 passed")
        x_fst = self.fstmod_out(x_fst)
        # print("decode 1 passed")
        x_snd = self.sndmod_out(x_snd)
        # print("decode 2 passed")

        return x_fst, x_snd

    def _classification_processing(
        self, latent_representation: Tensor, y: Tensor, compute_accuracy: bool
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        logits = self.classification_head(latent_representation)
        c_loss = F.cross_entropy(logits, y, weight=self.cfg.class_weights)
        if compute_accuracy:
            acc = ((torch.argmax(logits, dim=1) == y).float().mean()).item()
            bac = balanced_accuracy_score(
                y.cpu().numpy(), torch.argmax(logits, dim=1).cpu().numpy()
            )
            return c_loss, acc, bac

        return c_loss, None, None

    def forward(
        self, batch: Tuple[Tensor], compute_accuracy: bool = False
    ) -> Tuple[Tensor, Dict[str, float]]:
        metrics = {}
        total_loss = 0.0
        x_fst, x_snd, *packed_labels_potentially = batch
        assert isinstance(x_fst, Tensor), TypeError(
            f"x_fst must be a Tensor, got {type(x_fst)} instead."
        )
        assert isinstance(x_snd, Tensor), TypeError(
            f"x_snd must be a Tensor, got {type(x_snd)} instead."
        )
        z = self._encode(x_fst, x_snd)
        assert isinstance(z, Tensor), TypeError(
            f"z must be a Tensor, got {type(z)} instead."
        )
        x_fst_hat, x_snd_hat = self._decode(z)
        assert isinstance(x_fst_hat, Tensor), TypeError(
            f"x_fst_hat must be a Tensor, got {type(x_fst_hat)} instead."
        )
        assert isinstance(x_snd_hat, Tensor), TypeError(
            f"x_snd_hat must be a Tensor, got {type(x_snd_hat)} instead."
        )
        recon_loss = F.mse_loss(x_fst_hat, x_fst) + F.mse_loss(x_snd_hat, x_snd)
        metrics["recon_loss"] = recon_loss.item()
        total_loss += self.cfg.recon_loss_coef * recon_loss

        if self.cfg.classification_head:
            c_loss, acc, bac = self._classification_processing(
                z, *packed_labels_potentially, compute_accuracy
            )
            metrics["class_loss"] = c_loss.item()
            if acc is not None:
                metrics["acc"] = acc
                metrics["bac"] = bac
            total_loss += self.cfg.c_loss_coef * c_loss

        return total_loss, metrics

    def training_step(self, batch: Tensor) -> Tensor:
        loss, loss_components = self(batch)
        self.log("Train loss", loss, on_epoch=True, prog_bar=True)
        for k, v in loss_components.items():
            self.log(f"Train {k}", v, on_epoch=True, prog_bar=True)

    def validation_step(self, batch: Tensor) -> Tensor:
        loss, loss_components = self(batch)
        self.log("Val loss", loss, on_epoch=True, prog_bar=True)
        for k, v in loss_components.items():
            self.log(f"Val {k}", v, on_epoch=True, prog_bar=True)

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

    def _get_decoder_jacobian(self, z: Tensor) -> Tensor:
        return torch.autograd.functional.jacobian(self.decoder, z)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)

    def assert_cfg(self, cfg: Namespace) -> None:
        default_cfg = {
            "max_epochs": 5,
            "log_every_n_steps": 1,
            "first_modality_hidden_dim": 50,
            "second_modality_hidden_dim": 10,
            "first_modality_embedding_dim": 50,
            "second_modality_embedding_dim": 10,
            "encoder_hidden_dim": 5000,
            "encoder_out_dim": 40,
            "decoder_in_dim": 10,
            "decoder_hidden_dim": 20,
            "recon_loss_coef": 1,
            "c_loss_coef": 1,
            "kld_loss_coef": 1,
            "lr": 0.001,
        }
        for attr, default_value in default_cfg.items():
            if not hasattr(cfg, attr):
                setattr(cfg, attr, default_value)
                print(f"{attr} set as {default_value}")


class OmiGMPriorProbabilisticAE(OmiAE):
    def __init__(self, cfg: Namespace):
        super(OmiGMPriorProbabilisticAE, self).__init__(cfg)
        self.assert_cfg(cfg)
        self.component_logits = nn.Parameter(
            data=torch.zeros(size=(cfg.no_components,)), requires_grad=True
        )
        self.means = nn.Parameter(
            torch.randn(cfg.no_components, cfg.latent_dim), requires_grad=True
        )
        # STDs of GMM
        self.register_buffer(
            "stds", cfg.components_std * torch.ones(cfg.no_components, cfg.latent_dim)
        )

    def _var_transformation(self, logvar: Tensor) -> Tensor:
        return torch.exp(0.5 * logvar)

    def forward(
        self, batch: Tuple[Tensor], compute_accuracy: bool = False
    ) -> Tuple[Tensor]:
        metrics: Dict[str, float] = {}
        total_loss: Tensor = 0.0
        (
            x_fst,
            x_snd,
            labels,
        ) = batch  # ASSUMPTION THAT ALL LABELS ARE AVAILABLE (the extension to the mix of alebeled + unlabeled is not difficuls, but it is not implemented here as it may not be necessary for the task at hand)
        labels = torch.bernoulli(torch.ones_like(labels) * 0.5).long()
        # assert (
        #     False
        # ), "The labels are random for now, this should be changed to the actual labels"
        z_means, z_stds = self._encode(x_fst, x_snd).chunk(2, dim=1)
        z_stds = self._var_transformation(z_stds)
        normal_rv = self._make_normal_rv(z_means, z_stds)
        entropy_per_batch_sample = normal_rv.entropy().sum(dim=1).unsqueeze(0)  # [1, B]
        assert entropy_per_batch_sample.shape == (1, x_fst.shape[0]), AssertionError(
            f"Entropy shape is {entropy_per_batch_sample.shape}, expected {(1, x_fst.shape[0])}"
        )
        z_sample = normal_rv.rsample(
            sample_shape=(self.cfg.no_latent_samples,)
        ).unsqueeze(
            2
        )  # [K, B, 1, latent_dim]
        assert z_sample.shape == (
            self.cfg.no_latent_samples,
            x_fst.shape[0],
            1,
            self.cfg.latent_dim,
        ), AssertionError(
            f"z_sample shape is {z_sample.shape}, expected {(self.cfg.no_latent_samples, x_fst.shape[0], 1, self.cfg.latent_dim)}"
        )

        gmm = self._make_gmm()
        per_component_logprob = gmm.component_distribution.log_prob(
            z_sample
        )  # [K, B, no_components]
        assert per_component_logprob.shape == (
            self.cfg.no_latent_samples,
            x_fst.shape[0],
            self.cfg.no_components,
        ), AssertionError(
            f"per_component_logprob shape is {per_component_logprob.shape}, expected {(self.cfg.no_latent_samples, x_fst.shape[0], self.cfg.no_components)}"
        )
        component_indicator = torch.arange(self.cfg.no_components).unsqueeze(0).repeat(
            (x_fst.shape[0], 1)
        ) == labels.unsqueeze(1)
        assert component_indicator.shape == (
            x_fst.shape[0],
            self.cfg.no_components,
        ), AssertionError(
            f"component_indicator shape is {component_indicator.shape}, expected {(x_fst.shape[0], self.cfg.no_components)}"
        )
        gmm_likelihood_per_k = per_component_logprob[:, component_indicator]  # [K, B]
        assert gmm_likelihood_per_k.shape == (
            self.cfg.no_latent_samples,
            x_fst.shape[0],
        ), AssertionError(
            f"gmm_likelihood_per_k shape is {gmm_likelihood_per_k.shape}, expected {(self.cfg.no_latent_samples, x_fst.shape[0])}"
        )

        x_fst_hat, x_snd_hat = self._decode(z_sample.squeeze(2))
        recon_loss_per_k = F.mse_loss(
            x_fst_hat, x_fst.repeat(self.cfg.no_latent_samples, 1, 1), reduction="none"
        ).mean(dim=-1) + F.mse_loss(
            x_snd_hat, x_snd.repeat(self.cfg.no_latent_samples, 1, 1), reduction="none"
        ).mean(
            dim=-1
        )  # [K, B]
        assert recon_loss_per_k.shape == (
            self.cfg.no_latent_samples,
            x_fst.shape[0],
        ), AssertionError(
            f"recon_loss_per_k shape is {recon_loss_per_k.shape}, expected {(self.cfg.no_latent_samples, x_fst.shape[0])}"
        )

        if self.cfg.no_latent_samples > 1:  # IWAE with no_latent_samples latent samples
            total_loss = -torch.logsumexp(
                # gmm_likelihood_per_k + recon_loss_per_k + entropy_per_batch_sample,
                self.cfg.gmm_likelihood_loss_coef * gmm_likelihood_per_k
                + self.cfg.entropy_loss_coef * entropy_per_batch_sample
                + self.cfg.recon_loss_coef * recon_loss_per_k,
                dim=0,
            ).mean()
        else:  # IWAE reduces to VAE with one latent sample
            total_loss = -(
                gmm_likelihood_per_k + recon_loss_per_k + entropy_per_batch_sample
            ).mean()

        metrics["entropy"] = entropy_per_batch_sample.mean().item()
        metrics["gmm_likelihood"] = gmm_likelihood_per_k.mean().item()
        metrics["recon_loss"] = recon_loss_per_k.mean().item()

        if self.cfg.classification_head:
            c_loss, acc, bac = self._classification_processing(
                z_means, labels, compute_accuracy
            )
            metrics["class_loss"] = c_loss.item()
            if acc is not None:
                metrics["acc"] = acc
                metrics["bac"] = bac
            total_loss += self.cfg.c_loss_coef * c_loss

        return total_loss, metrics

    def _make_normal_rv(self, mu: Tensor, logvar: Tensor):
        return td.Normal(mu, self._var_transformation(logvar))

    def _make_gmm(self):
        categorical = td.Categorical(logits=self.component_logits)
        comp = td.Independent(
            td.Normal(self.means, self.stds), reinterpreted_batch_ndims=1
        )
        return td.MixtureSameFamily(categorical, comp)

    def assert_cfg(self, cfg: Namespace) -> None:
        super(OmiGMPriorProbabilisticAE, self).assert_cfg(cfg)
        default_cfg = {
            "no_components": 2,
            "components_std": 1,
            "no_latent_samples": 16,
            "gmm_likelihood_loss_coef": 0.1,
            "entropy_loss_coef": 0.1,
        }
        for attr, default_value in default_cfg.items():
            if not hasattr(cfg, attr):
                setattr(cfg, attr, default_value)
                print(f"{attr} set as {default_value}")

        assert cfg.latent_dim * 2 == cfg.encoder_out_dim, ValueError(
            "The latent dimension must be twice the encoder output dimension"
        )


# class OmiHierarchicalGMPriorProbabilisticAE(OmiGMPriorProbabilisticAE): :) to be continued...

# class M1M2 MODEL :) to be continued...

# CLASS NALEŚNIK MODEL IF IT EVEN DIFFERES FROM THE ABOVE MODELS :) to be continued...

_OMIVAE_IMPLEMENTATIONS = {
    "OmiAE": OmiAE,
    "OmiGMPriorProbabilisticAE": OmiGMPriorProbabilisticAE,
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
                batch_size=self.cfg.batch_size,
                shuffle=True,
                first_modality_dim=self.cfg.first_modality_dim,
                second_modality_dim=self.cfg.second_modality_dim,
                include_class_labels=self.cfg.classification_head
                or self.cfg.include_class_labels,
                target_hierarchy_level=self.cfg.target_hierarchy_level,
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
        pass

    def predict_proba(self, data: AnnData):
        pass

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
        assert cfg.omivae_implementation in _OMIVAE_IMPLEMENTATIONS, ValueError(
            f"Invalid OmiVAE implementation: {cfg.omivae_implementation}"
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

        if not hasattr(cfg, "early_stopping"):
            setattr(cfg, "early_stopping", True)
            print(f"early_stopping set as True")
        if cfg.early_stopping:
            if not hasattr(cfg, "min_delta"):
                setattr(cfg, "min_delta", 0.001)
                print(f"min_delta set as 0.001")
            if not hasattr(cfg, "patience"):
                setattr(cfg, "patience", 5)
                print(f"patience set as 5")


# class OmiVAE(OmiAE):

# class OmiIWAE(OmiAE):
