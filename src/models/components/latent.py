import abc
from argparse import Namespace
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, TypeAlias, TypeVar

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.loss import (
    StructuredLoss,
    format_loss,
    get_explicit_constraint,
    map_loss_name,
)

_EPS: float = 1e-8

Batch: TypeAlias = Dict[str, torch.Tensor]

StructuredForwardOutput: TypeAlias = Dict[str, Batch | List[StructuredLoss]]

GaussianRV: TypeAlias = td.Normal

GaussianMixtureRV: TypeAlias = (
    td.MixtureSameFamily
)  # [td.Categorical, GaussianRV] - complex typing does not work


def _format_forward_output(
    batch: Batch, losses: List[StructuredLoss]
) -> StructuredForwardOutput:
    return {"batch": batch, "losses": losses}


def make_normal_rv(mean: torch.Tensor, std: torch.Tensor) -> GaussianRV:
    return td.Normal(loc=mean, scale=std)


def make_gm_rv(
    component_logits: torch.Tensor, means: torch.Tensor, std: float
) -> GaussianMixtureRV:
    n_components, dim = means.shape
    categorical = td.Categorical(logits=component_logits)
    stds = std * torch.ones(n_components, dim)
    comp = td.Independent(td.Normal(loc=means, scale=stds), reinterpreted_batch_ndims=1)
    return td.MixtureSameFamily(categorical, comp)


def _sample_nondiff(
    rv: td.distribution.Distribution, sample_shape: torch.Size | Tuple[int]
) -> torch.Tensor:
    return rv.sample(sample_shape=sample_shape)


def _sample_diff_gm_rv(
    rg_rv: GaussianMixtureRV, sample_shape: torch.Size
) -> torch.Tensor:
    logits_shape = rg_rv.mixture_distribution.logits.shape
    repeated_logits = rg_rv.mixture_distribution.logits.view(
        *(1 for _ in range(len(sample_shape))), *logits_shape
    ).repeat(*sample_shape, 1)
    sampled_components_one_hots: torch.Tensor = F.gumbel_softmax(
        repeated_logits,
        dim=1,
        hard=True,
    )  # (*sample_shape, n_components)
    component_samples: torch.Tensor = rg_rv.component_distribution.rsample(
        sample_shape=sample_shape
    )  # (*sample_shape, n_components, dim)
    selected_component_samples = (
        sampled_components_one_hots[:, :, None] * component_samples
    ).sum(
        dim=1
    )  # (*sample_shape, dim)
    return selected_component_samples


# def calc_class_prob_from_gm_sample(rv: GaussianMixtureRV)

_STD_TRANSFORMS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "softplus": lambda x: F.softplus(x) + _EPS
}


def _get_std_transform(transform_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    return _STD_TRANSFORMS[transform_name]


class GaussianPosterior(nn.Module):  # nn.Module for compatibility
    def __init__(self, cfg: Namespace) -> None:
        super(GaussianPosterior, self).__init__()
        self._std_transformation: Callable = _get_std_transform(cfg.std_transformation)

        self._data_name: str = cfg.data_name

    def forward(self, batch: Batch) -> StructuredForwardOutput:
        means, stds = batch[self._data_name].chunk(chunks=2, dim=1)
        stds = self._std_transformation(stds)
        rv = make_normal_rv(mean=means, std=stds)
        entropy_per_batch_sample = (
            rv.entropy().sum(dim=-1).unsqueeze(0)
        )  # (1, batch_size)
        # Replacing the 'data' with reparametrisation trick samples in the batch and leaving the rest unchanged.
        batch[self._data_name] = rv.rsample(
            sample_shape=(self.cfg.no_latent_samples,)
        ).unsqueeze(
            2
        )  # (n_latent_samples, batch_size, 1, latent_dim)
        return _format_forward_output(
            batch=batch,
            losses=[
                format_loss(
                    loss=entropy_per_batch_sample,
                    name=map_loss_name(loss_name="posterior_entropy"),
                    aggregated=False,
                )
            ],
        )


class _GM(nn.Module, abc.ABC):
    def __init__(self, cfg: Namespace) -> None:
        super(_GM, self).__init__()
        self._cfg = cfg
        # Learnable logits of GM components
        self._component_logits = nn.Parameter(
            data=torch.zeros(size=(cfg.n_components,)), requires_grad=True
        )
        # learnable means of GM
        self._component_means = nn.Parameter(
            torch.randn(cfg.n_components, cfg.latent_dim), requires_grad=True
        )
        torch.nn.init.xavier_normal
        # fixed STDs of GMM
        self.register_buffer("_std", cfg.components_std)
        self._rv: Optional[td.MixtureSameFamily[td.Normal]] = None

    def set_rv(self) -> None:
        self._rv = make_gm_rv(
            component_logits=self._component_logits,
            means=self._component_means,
            std=self._std,
        )

    def reset_rv(self) -> None:
        self._rv = None

    def _get_nll(self, x: torch.Tensor) -> torch.Tensor:
        gm_nll_per_lat_sampl = -self._rv.log_prob(x)
        return gm_nll_per_lat_sampl

    def _get_component_conditioned_nll(
        self, x: torch.Tensor, component_indicator: torch.Tensor
    ) -> torch.Tensor:
        gm_nll_per_component = -self._rv.component_distribution.log_prob(
            x
        )  # (n_latent_samples, batch_size, n_components)
        gm_nll_lat_sampl = gm_nll_per_component[
            :, torch.arange(component_indicator.shape[0]), component_indicator
        ]  # (n_latent_samples, batch_size)
        return gm_nll_lat_sampl

    @abc.abstractmethod
    def forward(self, batch: Batch) -> StructuredForwardOutput:
        pass


class GMPriorNLL(_GM):
    def __init__(self, cfg: Namespace) -> None:
        super(GMPriorNLL, self).__init__(cfg=cfg)
        self._data_name: str = cfg.data_name

    def forward(self, batch: Batch) -> StructuredForwardOutput:
        x, component_indicator = (
            batch[self._data_name],
            batch[self._cfg.component_indicator],
        )
        self.set_rv()
        unknown_mask = (
            component_indicator == -1
        )  # -1 indicates that the component is unknown for a sample
        known_mask = unknown_mask.logical_not
        nll_lat_sampl = torch.zeros(
            (self._cfg.n_latent_samples, x.shape[0])
        )  # (n_latent_samples, batch_size)
        if any(unknown_mask):  # There are samples with unknown components.
            unknown_gm_nll_lat_sampl = self._get_nll(x=x[unknown_mask])
            nll_lat_sampl += unknown_gm_nll_lat_sampl
        if any(known_mask):  # There are samples with known components
            known_gm_nll_lat_sampl = self._get_component_conditioned_nll(
                x=x[known_mask],
                component_indicator=component_indicator[known_mask],
            )  # (n_latent_samples, known_batch_size)
            nll_lat_sampl += known_gm_nll_lat_sampl
        self.reset_rv()
        return _format_forward_output(
            batch=batch,
            losses=[
                format_loss(
                    loss=nll_lat_sampl,
                    name=map_loss_name(loss_name="prior_nll"),
                    aggregated=False,
                )
            ],
        )


class PairwiseDistances(NamedTuple):
    dist_xx: torch.Tensor
    dict_xy: torch.Tensor
    dict_yy: torch.Tensor


def calc_pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> PairwiseDistances:
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0)
    ry = yy.diag().unsqueeze(0)

    dist_xx = rx.t() + rx - 2.0 * xx
    dist_xy = rx.t() + ry - 2.0 * zz
    dist_yy = ry.t() + ry - 2.0 * yy

    return PairwiseDistances(dist_xx=dist_xx, dist_xy=dist_xy, dist_yy=dist_yy)


# _KERNELS:


def calc_mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""
    Compute the Maximum Mean Discrepancy (MMD) between two samples
    """
    n = x.shape[0]
    m = y.shape[0]

    dxx, dxy, dyy = calc_pairwise_distances(x=x, y=y)

    # TODO: make the way of establishing a hyperparameter
    # C = torch.median(dxy) # idk weather this is the right way to calculate the median heuristic
    C = 2 * self.cfg.model.latent_dim * self.cfg.model.latent.components_std
    if self.cfg.model.mmd.kernel == "inverse_multiquadric":
        XX = C / (dxx + C).sum()
        YY = C / (dyy + C).sum()
        XY = C / (dxy + C).sum()
    elif self.cfg.model.mmd.kernel == "rbf":
        XX = torch.exp(-0.5 * dxx / C).sum()
        YY = torch.exp(-0.5 * dyy / C).sum()
        XY = torch.exp(-0.5 * dxy / C).sum()

    denominator_xx = (
        n * (n - 1) if n != 1 else 1
    )  # sthg has to be done if we have only one sample, it seeems we are biased then though
    denominator_yy = m * (m - 1) if m != 1 else 1  # same as above

    return XX / denominator_xx + YY / denominator_yy - 2.0 * XY / (n * m)


class GMPriorMMD(_GM):
    def __init__(self, cfg: Namespace) -> None:
        super(GMPriorNLL, self).__init__(cfg=cfg)
        self._data_name: str = cfg.data_name

    def forward(self, batch: Batch) -> StructuredForwardOutput:
        x, component_indicator = (
            batch[self._data_name],
            batch[self._cfg.component_indicator],
        )
        self.set_rv()
        unknown_mask = (
            component_indicator == -1
        )  # -1 indicates that the component is unknown for a sample
        known_mask = unknown_mask.logical_not
        nll_lat_sampl = torch.zeros(
            (self._cfg.n_latent_samples, x.shape[0])
        )  # (n_latent_samples, batch_size)
        if any(unknown_mask):  # There are samples with unknown components.
            unknown_gm_nll_lat_sampl = self._get_nll(x=x[unknown_mask])
            nll_lat_sampl += unknown_gm_nll_lat_sampl
        if any(known_mask):  # There are samples with known components
            known_gm_nll_lat_sampl = self._get_component_conditioned_nll(
                x=x[known_mask],
                component_indicator=component_indicator[known_mask],
            )  # (n_latent_samples, known_batch_size)
            nll_lat_sampl += known_gm_nll_lat_sampl
        self.reset_rv()
        return _format_forward_output(
            batch=batch,
            losses=[
                format_loss(
                    loss=nll_lat_sampl,
                    name=map_loss_name(loss_name="prior_nll"),
                    aggregated=False,
                )
            ],
        )


# HierarchicalGMPrior - is it regular hierarchical model?


class LatentConstraint(nn.Module):  # nn.Module for compatibility
    def __init__(self, cfg: Namespace) -> None:
        super(LatentConstraint, self).__init__()
        self._calculate_constraint = get_explicit_constraint(
            constraint_name=cfg.constraint_method
        )
        self.constraint_method = cfg.constraint_method
        self._data_name: str = cfg.data_name

    def forward(self, batch: Batch) -> StructuredForwardOutput:
        batch[self._data_name] = self._constraint_method
        return _format_forward_output(
            batch=batch,
            losses=[
                format_loss(
                    loss=self._calculate_constraint(x=batch[self._data_name], dim=None),
                    name=map_loss_name(loss_name="latent_constraint"),
                    aggregated=True,
                )
            ],
        )


class FuzzyClustering(_GM):
    def __init__(self, cfg: Namespace) -> None:
        super(FuzzyClustering, self).__init__(cfg=cfg)
        self._calculate_constraint = get_explicit_constraint(
            constraint_name=cfg.constraint_method
        )
        self._data_name: str = cfg.data_name

    def _calculate_component_constraint(self) -> torch.Tensor:
        # Component mean regularization bringing the components' means. Weighted by the components' probabilities.
        component_regularization = (
            self._calculate_constraint(x=self._component_means, dim=1)
            * torch.softmax(self._component_logits).detach()
        ).sum()  # (1)
        return component_regularization

    def forward(self, batch: Batch) -> StructuredForwardOutput:
        self.set_rv()
        gm_nll = self._get_nll(x=batch[self._data_name])  # (1)
        comp_reg = self._calculate_component_constraint()
        self.reset_rv()
        return _format_forward_output(
            batch=batch,
            losses=[
                format_loss(
                    loss=gm_nll, name=map_loss_name(loss_name="latent_fuzzy_clustering")
                ),
                format_loss(
                    loss=comp_reg,
                    name=map_loss_name(loss_name="clustering_component_reg"),
                ),
            ],
        )


# z_to_decode = z_sample.squeeze(2).reshape(
#     -1, self.cfg.latent_dim
# )  # (self.cfg.no_latent_samples * batch_size, self.cfg.latent_dim)
# # print("z_to_decode", z_to_decode.shape)
# x_hat = self.decoder(z_to_decode).reshape(
#     self.cfg.no_latent_samples, batch_size, -1
# )  # (self.cfg.no_latent_samples, batch_size, sum_of_modalities)
# # print("x_hat", x_hat.shape, x.repeat(self.cfg.no_latent_samples, 1, 1).shape)
# # assert x_hat.shape == x.repeat(self.cfg.no_latent_samples, 1, 1).shape
# recon_loss_lat_sampl = F.mse_loss(
#     x_hat, x.repeat(self.cfg.no_latent_samples, 1, 1), reduction="none"
# ).mean(
#     dim=-1
# )  # (no_laten_samples, batch_size)


# LATENT QUANTIZATION FROM VQVAE HOMEWORK
class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        beta: float = 0.25,
    ):
        super(VectorQuantizer, self).__init__()
        # TODO
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._beta = beta
        self._embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self._embeddings.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )

    def forward(self, inputs):
        BATCH_SIZE, _, H, W = inputs.shape

        distances = torch.cdist(
            inputs.permute((0, 2, 3, 1)).reshape(
                (BATCH_SIZE * H * W, self._embedding_dim)
            ),
            self._embeddings.weight,
        )  # BATCH_SIZE, C, H, W -> BATCH_SIZE, H, W, C - change for cdist
        matched_indices = torch.argmin(distances, dim=1).view((BATCH_SIZE * H * W, 1))
        quantized_latent = (
            self._embeddings(matched_indices)
            .view((BATCH_SIZE, H, W, self._embedding_dim))
            .permute((0, 3, 1, 2))
        )  # BATCH_SIZE, H, W, C -> BATCH_SIZE, C, H, W - change for comparison with inputs

        q_loss = F.mse_loss(
            quantized_latent, inputs.detach()
        )  # dictionary learning loss
        e_loss = F.mse_loss(quantized_latent.detach(), inputs)  # commitment loss
        vq_loss = q_loss + self._beta * e_loss  # VectorQuantizer loss

        quantized_latent = (
            inputs + (quantized_latent - inputs).detach()
        )  # this is the straight-through estimator. quantized_latent is detached so the gradient won't flow from the decoder to the embeddings

        return (
            quantized_latent,
            q_loss,
            e_loss,
            vq_loss,
            matched_indices.view((BATCH_SIZE, H, W)),
        )


# dVAE implementation, softmax weighted, and gumbel sampled
