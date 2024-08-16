import abc
from argparse import Namespace
from ctypes import Structure
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, TypeAlias, TypeVar
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum

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
    gm_rv: GaussianMixtureRV, sample_shape: torch.Size
) -> torch.Tensor:
    pattern = f'comps -> {" ".join(["1"] * len(sample_shape))} comps'
    repeated_logits = rearrange(
        tensor=gm_rv.mixture_distribution.logits, pattern=pattern
    ).repeat(*sample_shape, 1)
    sampled_components_one_hots: torch.Tensor = F.gumbel_softmax(
        repeated_logits,
        dim=1,
        hard=True,
    )  # (*sample_shape, n_components)
    component_samples: torch.Tensor = gm_rv.component_distribution.rsample(
        sample_shape=sample_shape
    )  # (*sample_shape, n_components, dim)
    selected_component_samples = (
        sampled_components_one_hots.unsqueeze(-1) * component_samples
    ).sum(
        dim=1
    )  # (*sample_shape, dim)
    return selected_component_samples


def calc_component_logits_for_gm_sample(
    rv: GaussianMixtureRV, x: torch.Tensor
) -> torch.Tensor:
    # the logits are calculated up to a shift due to logits.exp() not necessarily summing to 1 and the normalizing denominator p(x) which is ignored
    # p(C = c | x) = (p(x | C = c) * p(C = c)) / p(x)
    pattern = f'comps -> {" ".join(["1"] * len(x.shape[:-1]))} comps'
    prior_log_probs = rearrange(rv.mixture_distribution.logits, pattern=pattern)
    per_component_log_probs = rv.component_distribution.log_prob(
        x.unsqueeze(dim=-2)
    )  # (*x.shape[:-1], n_latent_components)
    return prior_log_probs + per_component_log_probs


_STD_TRANSFORMS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "softplus": lambda x: F.softplus(x) + _EPS
}


def _get_std_transform(transform_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    return _STD_TRANSFORMS[transform_name]


class GaussianPosterior(nn.Module):  # nn.Module for compatibility
    def __init__(self, cfg: Namespace, data_name: str) -> None:
        super(GaussianPosterior, self).__init__()
        self._std_transformation: Callable = _get_std_transform(cfg.std_transformation)
        self._data_name: str = data_name
        self._n_latent_samples: int = cfg.n_latent_samples

    def forward(self, batch: Batch) -> StructuredForwardOutput:
        means, stds = batch[self._data_name].chunk(chunks=2, dim=1)
        stds = self._std_transformation(stds)
        rv = make_normal_rv(mean=means, std=stds)
        entropy_per_batch_sample = einsum(rv.entropy(), "batch dim -> batch").unsqueeze(
            0
        )  # (1, batch_size)
        # Replacing the 'data' with reparametrisation trick samples in the batch and leaving the rest unchanged.
        latent_samples = rv.rsample(
            sample_shape=(self._n_latent_samples,)
        )  # (n_latent_samples, batch_size, latent_dim)
        batch[self._data_name] = rearrange(
            tensor=latent_samples, pattern="samp batch dim -> batch samp 1 dim"
        )
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
        # Learnable logits of GM components
        self._component_logits = nn.Parameter(
            data=torch.zeros(size=(cfg.n_components,)), requires_grad=True
        )
        # learnable means of GM
        self._component_means = nn.Parameter(
            torch.empty(cfg.n_components, cfg.latent_dim), requires_grad=True
        )
        torch.nn.init.xavier_normal_(self._component_means)
        # fixed STDs of GMM
        self.register_buffer(
            "_std", torch.tensor(cfg.components_std, dtype=torch.float32)
        )
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
        x = rearrange(x, "batch sampl dim -> batch sampl 1 dim")
        gm_nll_per_component = -self._rv.component_distribution.log_prob(
            x
        )  # (batch_size, n_latent_samples, n_components)
        gm_nll_per_lat_sampl = gm_nll_per_component[
            torch.arange(component_indicator.shape[0]), :, component_indicator
        ]  # (batch_size, n_latent_samples)
        return gm_nll_per_lat_sampl

    @abc.abstractmethod
    def forward(self, batch: Batch) -> StructuredForwardOutput:
        pass


class GMPriorNLL(_GM):

    def __init__(
        self, cfg: Namespace, data_name: str, component_indicator_name: str
    ) -> None:
        super(GMPriorNLL, self).__init__(cfg=cfg)
        self._data_name: str = data_name
        self._component_indicator_name: str = component_indicator_name

    def forward(self, batch: Batch) -> StructuredForwardOutput:
        x, component_indicator = (
            batch[self._data_name],
            batch[self._component_indicator_name],
        )
        # x (batch_size, n_latent_samples, dim)
        self.set_rv()
        unknown_mask = (
            component_indicator == -1
        )  # -1 indicates that the component is unknown for a sample
        known_mask = unknown_mask.logical_not()
        # unknown_idx = unknown_mask.nonzero()
        # known_idx = known_mask.nonzero()
        nll_lat_sampl = torch.zeros((x.shape[:2]))  # (batch_size, n_latent_samples)
        if unknown_mask.any():  # There are samples with unknown components.
            unknown_gm_nll_lat_sampl = self._get_nll(
                x=x[unknown_mask]
            )  # (unknown_batch_size, n_latent_samples)
            # nll_lat_sampl.scatter_(dim=, index=, src=unknown_gm_nll_lat_sampl)
            nll_lat_sampl[unknown_mask] = unknown_gm_nll_lat_sampl  #
        if known_mask.any():  # There are samples with known components.
            known_gm_nll_lat_sampl = self._get_component_conditioned_nll(
                x=x[known_mask],
                component_indicator=component_indicator[known_mask],
            )  # (known_batch_size, n_latent_samples)
            # assert known_gm_nll_lat_sampl.shape[0] == known_mask.sum()
            # assert known_gm_nll_lat_sampl.shape[1] == x.shape[1]
            # nll_lat_sampl += known_gm_nll_lat_sampl
            nll_lat_sampl[known_mask] = known_gm_nll_lat_sampl
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
    dist_xy: torch.Tensor
    dist_yy: torch.Tensor


def calc_pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> PairwiseDistances:
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0)
    ry = yy.diag().unsqueeze(0)

    dist_xx = rx.t() + rx - 2.0 * xx
    dist_xy = rx.t() + ry - 2.0 * zz
    dist_yy = ry.t() + ry - 2.0 * yy

    return PairwiseDistances(dist_xx=dist_xx, dist_xy=dist_xy, dist_yy=dist_yy)


Kernel: TypeAlias = Callable[[torch.Tensor, float], float]

_KERNELS: Dict[str, Kernel] = {
    "rbf": lambda x, C: (-0.5 * x / C).exp().sum(),
    "inverse_multiquadratic": lambda x, C: C / (x + C).sum(),
}


def _get_kernel(kernel_name: str) -> Kernel:
    return _KERNELS[kernel_name]


def calc_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_name: str,
    latent_dim: int,
    components_std: float,
) -> torch.Tensor:
    r"""
    Compute the Maximum Mean Discrepancy (MMD) between two samples
    """
    n = x.shape[0]
    m = y.shape[0]

    dxx, dxy, dyy = calc_pairwise_distances(x=x, y=y)

    # TODO: make the way of establishing a hyperparameter
    # C = torch.median(dxy) # idk weather this is the right way to calculate the median heuristic
    C = 2 * latent_dim * components_std
    kernel = _get_kernel(kernel_name=kernel_name)

    XX = kernel(dxx, C)
    YY = kernel(dyy, C)
    XY = kernel(dxy, C)

    denominator_xx = (
        n * (n - 1) if n != 1 else 1
    )  # sthg has to be done if we have only one sample, it seeems we are biased then though
    denominator_yy = m * (m - 1) if m != 1 else 1  # same as above

    return XX / denominator_xx + YY / denominator_yy - 2.0 * XY / (n * m)


class LatentConstraint(nn.Module):  # nn.Module for compatibility
    def __init__(self, cfg: Namespace, data_name: str) -> None:
        super(LatentConstraint, self).__init__()
        self._calculate_constraint = get_explicit_constraint(
            constraint_name=cfg.constraint_method
        )
        self.constraint_method = cfg.constraint_method
        self._data_name: str = data_name

    def forward(self, batch: Batch) -> StructuredForwardOutput:
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
    def __init__(self, cfg: Namespace, data_name: str) -> None:
        super(FuzzyClustering, self).__init__(cfg=cfg)
        self._calculate_constraint = get_explicit_constraint(
            constraint_name=cfg.constraint_method
        )
        self._data_name: str = data_name

    def _calculate_component_constraint(self) -> torch.Tensor:
        # Component mean regularization bringing the components' means. Weighted by the components' probabilities.
        component_regularization = (
            self._calculate_constraint(x=self._component_means, dim=1)
            * torch.softmax(self._component_logits.detach(), dim=0)
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
                    loss=gm_nll,
                    name=map_loss_name(loss_name="latent_fuzzy_clustering"),
                    aggregated=True,
                ),
                format_loss(
                    loss=comp_reg,
                    name=map_loss_name(loss_name="clustering_component_reg"),
                    aggregated=True,
                ),
            ],
        )


# HierarchicalGMPrior - is it regular hierarchical model?
# every level of a hierarchy is specified based on the on lowest level labels

# class GMPriorMMD(_GM): - requires some code workaround code to get working with trainable prior params - possible with differentiable sampling from gaussian mixture.


# # LATENT QUANTIZATION FROM VQVAE HOMEWORK
# class VectorQuantizer(nn.Module):
#     def __init__(
#         self,
#         cfg: Namespace,
#         data_name: str
#     ):
#         super(VectorQuantizer, self).__init__()
#         self._data_name: str = data_name
#         self._num_embeddings: int = cfg.num_embeddings
#         self._embedding_dim: int = cfg.embedding_dim
#         self._embeddings = nn.Embedding(cfg.num_embeddings, cfg.embedding_dim)
#         torch.nn.init.uniform_(self._embeddings.weight.data, a=-1 / self._num_embeddings, b=1 / self._num_embeddings)


#     def forward(self, batch: Batch) -> StructuredForwardOutput:
#         # BATCH_SIZE, _, H, W = inputs.shape
#         # distances = rearrange()
#         rearrange(batch[self._data_name], "batch dim")
#         distances = torch.cdist(
#             inputs.permute((0, 2, 3, 1)).reshape(
#                 (BATCH_SIZE * H * W, self._embedding_dim)
#             ),
#             self._embeddings.weight,
#         )  # BATCH_SIZE, C, H, W -> BATCH_SIZE, H, W, C - change for cdist
#         matched_indices = torch.argmin(distances, dim=1).view((BATCH_SIZE * H * W, 1))
#         quantized_latent = (
#             self._embeddings(matched_indices)
#             .view((BATCH_SIZE, H, W, self._embedding_dim))
#             .permute((0, 3, 1, 2))
#         )  # BATCH_SIZE, H, W, C -> BATCH_SIZE, C, H, W - change for comparison with inputs

#         q_loss = F.mse_loss(
#             quantized_latent, inputs.detach()
#         )  # dictionary learning loss
#         e_loss = F.mse_loss(quantized_latent.detach(), inputs)  # commitment loss
#         vq_loss = q_loss + self._beta * e_loss  # VectorQuantizer loss

#         quantized_latent = (
#             inputs + (quantized_latent - inputs).detach()
#         )  # this is the straight-through estimator. quantized_latent is detached so the gradient won't flow from the decoder to the embeddings

#         return (
#             quantized_latent,
#             q_loss,
#             e_loss,
#             vq_loss,
#             matched_indices.view((BATCH_SIZE, H, W)),
#         )


# dVAE implementation, softmax weighted, and gumbel sampled
