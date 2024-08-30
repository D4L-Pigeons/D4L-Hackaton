import abc
from argparse import Namespace
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, TypeAlias, Type
from flask import config
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from src.utils.common_types import (
    Batch,
    StructuredForwardOutput,
    format_structured_forward_output,
    format_structured_loss,
    ConfigStructure,
)

from src.models.components.loss import get_explicit_constraint, map_loss_name

from src.utils.config import validate_config_structure

_EPS: float = 1e-8


GaussianRV: TypeAlias = td.Normal

GaussianMixtureRV: TypeAlias = (
    td.MixtureSameFamily
)  # [td.Categorical, GaussianRV] - complex typing does not work


def make_normal_rv(mean: torch.Tensor, std: torch.Tensor) -> GaussianRV:
    r"""
    Create a Gaussian random variable with the given mean and standard deviation.

    Args:
        mean: The mean of the Gaussian random variable.
        std: The standard deviation of the Gaussian random variable.

    Returns:
        GaussianRV: The created Gaussian random variable.

    """
    return td.Normal(loc=mean, scale=std)


def make_gm_rv(
    component_logits: torch.Tensor, means: torch.Tensor, std: float
) -> GaussianMixtureRV:
    r"""
    Create a Gaussian Mixture Random Variable.

    Parameters:
        component_logits: Logits representing the mixture components.
        means: Mean values for each component.
        std: Standard deviation for all components.

    Returns:
        GaussianMixtureRV: A Gaussian Mixture Random Variable.

    """
    n_components, dim = means.shape
    categorical = td.Categorical(logits=component_logits)
    stds = std * torch.ones(n_components, dim)
    comp = td.Independent(td.Normal(loc=means, scale=stds), reinterpreted_batch_ndims=1)
    return td.MixtureSameFamily(categorical, comp)


def _sample_nondiff(
    rv: td.distribution.Distribution, sample_shape: torch.Size | Tuple[int]
) -> torch.Tensor:
    r"""
    Sample from a non-differentiable random variable.

    Args:
        rv: The random variable to sample from.
        sample_shape: The shape of the samples to generate.

    Returns:
        torch.Tensor: The samples generated from the random variable.
    """
    return rv.sample(sample_shape=sample_shape)


def _sample_diff_gm_rv(
    gm_rv: GaussianMixtureRV, sample_shape: torch.Size
) -> torch.Tensor:
    r"""
    Samples from a Gaussian Mixture Random Variable in a differentiable manner.

    Args:
        gm_rv: The Gaussian Mixture Random Variable object.
        sample_shape: The shape of the samples to be generated.

    Returns:
        torch.Tensor: The selected component samples from the GM-RV.

    """
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
    r"""
    Calculate the component logits for a Gaussian Mixture sample precisely up to a scalar shift
    due to logits.exp() not necessarily summing up to 1 and ignorance of normalizing denominator p(x).
    The whole formula is:
        log p(C = c | x) = log p(x | C = c) + log p(C = c) - log p(x)

    Args:
        rv (GaussianMixtureRV): The Gaussian Mixture random variable.
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The component logits.

    """

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
    std_transform = _STD_TRANSFORMS.get(transform_name, None)
    if std_transform is not None:
        return std_transform
    raise ValueError(
        f'The provided transform_name {transform_name} is wrong. Must be one of {" ,".join(list(_STD_TRANSFORMS.keys()))}'
    )


class GaussianPosterior(nn.Module):  # nn.Module for compatibility
    r"""
    Module for computing the posterior distribution of a Gaussian random variable.

    Args:
        cfg: Configuration object.

    Attributes:
        _std_transformation: Callable for transforming the standard deviations.
        _data_name: Name of the data in the batch.
        _n_latent_samples: Number of latent samples.

    Methods:
        forward(batch: Batch) -> StructuredForwardOutput:
            Computes the forward pass of the module.

    """

    _cfg_structure: ConfigStructure = {
        "std_transform": str,
        "data_name": str,
        "n_latent_samples": int,
        "loss_coef_posterior_entropy": float,
    }

    def __init__(self, cfg: Namespace) -> None:
        super(GaussianPosterior, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._cfg_structure)

        self._std_transform: Callable = _get_std_transform(cfg.std_transform)
        self._data_name: str = cfg.data_name
        self._n_latent_samples: int = cfg.n_latent_samples
        self._loss_coef_posterior_entropy: float = cfg.loss_coef_posterior_entropy

    def sample(self, batch: Batch) -> StructuredForwardOutput:
        means, stds = batch[self._data_name].chunk(chunks=2, dim=1)
        stds = self._std_transform(stds)
        rv = make_normal_rv(mean=means, std=stds)
        # Replacing the 'data' with reparametrisation trick samples in the batch and leaving the rest unchanged.
        latent_samples = rv.rsample(sample_shape=(1,))  # (1, batch_size, latent_dim)
        batch[self._data_name] = rearrange(
            tensor=latent_samples, pattern="samp batch dim -> batch samp dim"
        )
        return format_structured_forward_output(
            batch=batch,
        )

    def forward(self, batch: Batch) -> StructuredForwardOutput:
        r"""
        Input data shape: (batch dim)
        Output data shape: (batch sampl dim//2)
        """
        means, stds = batch[self._data_name].chunk(chunks=2, dim=1)
        stds = self._std_transform(stds)
        rv = make_normal_rv(mean=means, std=stds)
        entropy_per_batch_sample = einsum(rv.entropy(), "batch dim -> batch").unsqueeze(
            0
        )  # (1, batch_size)
        # Replacing the 'data' with reparametrisation trick samples in the batch and leaving the rest unchanged.
        latent_samples = rv.rsample(
            sample_shape=(self._n_latent_samples,)
        )  # (n_latent_samples, batch_size, latent_dim)
        batch[self._data_name] = rearrange(
            tensor=latent_samples, pattern="samp batch dim -> batch samp dim"
        )
        return format_structured_forward_output(
            batch=batch,
            losses=[
                format_structured_loss(
                    loss=entropy_per_batch_sample,
                    coef=self._loss_coef_posterior_entropy,
                    name=map_loss_name(loss_name="posterior_entropy"),
                    aggregated=False,
                )
            ],
        )


class _GM(nn.Module, abc.ABC):
    r"""
    Abstract base class for Gaussian Mixture (GM) models.

    Args:
        cfg: Configuration object containing model parameters.

    Attributes:
        _component_logits (nn.Parameter): Learnable logits of GM components.
        _component_means (nn.Parameter): Learnable means of GM.
        _std (torch.Tensor): Fixed standard deviations of GMM.
        _rv (Optional[td.MixtureSameFamily[td.Normal]]): Random variable representing the GM distribution.

    Methods:
        set_rv(): Sets the random variable representing the GM distribution.
        reset_rv(): Resets the random variable representing the GM distribution.
        _get_nll(x: torch.Tensor) -> torch.Tensor: Computes the negative log-likelihood of the GM model for a given input tensor.
        _get_component_conditioned_nll(x: torch.Tensor, component_indicator: torch.Tensor) -> torch.Tensor: Computes the negative log-likelihood of the GM model conditioned on a specific component indicator.
        forward(batch: Batch) -> StructuredForwardOutput: Abstract method to be implemented by subclasses for forward pass computation.

    """

    _config_structure: ConfigStructure = {
        "n_components": int,
        "latent_dim": int,
        "components_std": float,
    }

    def __init__(self, cfg: Namespace) -> None:
        super(_GM, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._component_logits = nn.Parameter(
            data=torch.zeros(size=(cfg.n_components,)), requires_grad=True
        )
        self._component_means = nn.Parameter(
            torch.empty(cfg.n_components, cfg.latent_dim), requires_grad=True
        )
        torch.nn.init.xavier_normal_(self._component_means)
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

    def sample(self, n_samples: int) -> Batch:
        if self._rv is None:
            raise RuntimeError(
                "Random variable not set. Call set_rv() before sampling."
            )

        component_samples = _sample_diff_gm_rv(self._rv, sample_shape=(n_samples,))
        batch = {
            self._data_name: rearrange(
                tensor=component_samples, pattern="samp batch dim -> batch samp dim"
            )
        }
        return format_structured_forward_output(batch=batch)

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


class GaussianMixturePriorNLL(_GM):
    _config_structure: ConfigStructure = {
        "data_name": str,
        "component_indicator_name": str,
        "loss_coef_prior_nll": float,
    } | _GM._config_structure  # Adding the config structure of the parent class.

    def __init__(self, cfg: Namespace) -> None:
        super(GaussianMixturePriorNLL, self).__init__(cfg=cfg)
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._data_name: str = cfg.data_name
        self._component_indicator_name: str = cfg.component_indicator_name
        self._loss_coef_prior_nll: float = cfg.loss_coef_prior_nll

    def forward(self, batch: Batch) -> StructuredForwardOutput:
        r"""
        Forward pass of the GaussianMixturePriorNLL model.

        Args:
            batch (Batch): Input batch containing data and component indicators.

        Returns:
            StructuredForwardOutput: Structured output containing the calculated loss.
        """
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
        nll_lat_sampl = torch.zeros((x.shape[:2]))  # (batch_size, n_latent_samples)
        if unknown_mask.any():  # There are samples with unknown components.
            unknown_gm_nll_lat_sampl = self._get_nll(
                x=x[unknown_mask]
            )  # (unknown_batch_size, n_latent_samples)
            assert (
                nll_lat_sampl[unknown_mask].shape == unknown_gm_nll_lat_sampl.shape
            ), f"nll_lat_sampl[unknown_mask].shape = {nll_lat_sampl[unknown_mask].shape} and unknown_gm_nll_lat_sampl.shape = {unknown_gm_nll_lat_sampl.shape}"
            nll_lat_sampl[unknown_mask] = unknown_gm_nll_lat_sampl  #
        if known_mask.any():  # There are samples with known components.
            known_gm_nll_lat_sampl = self._get_component_conditioned_nll(
                x=x[known_mask],
                component_indicator=component_indicator[known_mask],
            )  # (known_batch_size, n_latent_samples)
            nll_lat_sampl[known_mask] = known_gm_nll_lat_sampl
        self.reset_rv()
        return format_structured_forward_output(
            batch=batch,
            losses=[
                format_structured_loss(
                    loss=nll_lat_sampl,
                    coef=self._loss_coef_prior_nll,
                    name=map_loss_name(loss_name="prior_nll"),
                    aggregated=False,
                )
            ],
        )


class FuzzyClustering(_GM):
    r"""
    FuzzyClustering class for performing fuzzy clustering in the latent space.

    This class inherits from the _GM class and implements the functionality for fuzzy clustering. It calculates the component mean regularization and performs a forward pass to obtain the output of the fuzzy clustering.

    Attributes:
        _config_structure (ConfigStructure): The configuration structure for the FuzzyClustering class.

    Methods:
        __init__(self, cfg: Namespace): Initializes the FuzzyClustering object with the given configuration.
        _calculate_component_constraint(self) -> torch.Tensor: Calculates the component mean regularization.
        forward(self, batch: Batch) -> StructuredForwardOutput: Performs a forward pass of the FuzzyClustering class.
    """

    _config_structure: ConfigStructure = {
        "constraint_method": str,
        "data_name": str,
        "loss_coef_latent_fuzzy_clustering": float,
        "loss_coef_clustering_component_reg": float,
    } | _GM._config_structure

    def __init__(self, cfg: Namespace) -> None:
        super(FuzzyClustering, self).__init__(cfg=cfg)
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._calculate_constraint = get_explicit_constraint(
            constraint_name=cfg.constraint_method
        )
        self._data_name: str = cfg.data_name
        self._loss_coef_latent_fuzzy_clustering: float = (
            cfg.loss_coef_latent_fuzzy_clustering
        )
        self._loss_coef_clustering_component_reg: float = (
            cfg.loss_coef_clustering_component_reg
        )

    def _calculate_component_constraint(self) -> torch.Tensor:
        # Component mean regularization bringing the components' means. Weighted by the components' probabilities.
        component_regularization = (
            self._calculate_constraint(x=self._component_means, dim=1)
            * torch.softmax(self._component_logits.detach(), dim=0)
        ).sum()  # (1)
        return component_regularization

    def forward(self, batch: Batch) -> StructuredForwardOutput:
        r"""
        Input data shape: (batch dim)
        Output data shape: (batch dim)
        """
        self.set_rv()
        gm_nll = self._get_nll(x=batch[self._data_name])  # (1)
        comp_reg = self._calculate_component_constraint()
        self.reset_rv()
        return format_structured_forward_output(
            batch=batch,
            losses=[
                format_structured_loss(
                    loss=gm_nll,
                    coef=self._loss_coef_latent_fuzzy_clustering,
                    name=map_loss_name(loss_name="latent_fuzzy_clustering"),
                    aggregated=False,
                ),
                format_structured_loss(
                    loss=comp_reg,
                    coef=self._loss_coef_clustering_component_reg,
                    name=map_loss_name(loss_name="clustering_component_reg"),
                    aggregated=True,
                ),
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
    kernel = _KERNELS.get(kernel_name, None)
    if kernel is not None:
        return kernel
    raise ValueError(
        f'The provided kernel_name {kernel_name} is wrong. Must be one of {" ,".join(list(_KERNELS.keys()))}'
    )


def calc_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_name: str,
    latent_dim: int,
    components_std: float,
) -> torch.Tensor:
    r"""
    Compute the Maximum Mean Discrepancy (MMD) between two samples.

    Args:
        x (torch.Tensor): The first sample.
        y (torch.Tensor): The second sample.
        kernel_name (str): The name of the kernel function to be used.
        latent_dim (int): The dimension of the latent space.
        components_std (float): The standard deviation of the components.

    Returns:
        torch.Tensor: The MMD between the two samples.
    """
    n, m = x.shape[0], y.shape[0]

    dxx, dxy, dyy = calc_pairwise_distances(x=x, y=y)

    # TODO: make the way of establishing a hyperparameter
    # C = torch.median(dxy) # idk weather this is the right way to calculate the median heuristic
    C = 2 * latent_dim * components_std
    kernel = _get_kernel(kernel_name=kernel_name)

    XX, XY, YY = kernel(dxx, C), kernel(dxy, C), kernel(dyy, C)

    denominator_xx = (
        n * (n - 1) if n != 1 else 1
    )  # sthg has to be done if we have only one sample, it seeems we are biased then though
    denominator_yy = m * (m - 1) if m != 1 else 1  # same as above

    return XX / denominator_xx + YY / denominator_yy - 2.0 * XY / (n * m)


class LatentConstraint(nn.Module):  # nn.Module for compatibility
    r"""
    LatentConstraint is a class that represents a latent constraint module.

    Args:
        cfg (Namespace): The configuration object containing the constraint method.
        data_name (str): The name of the data used for calculating the constraint.

    Methods:
        forward(batch: Batch) -> StructuredForwardOutput:
            Performs the forward pass of the latent constraint module.

    Returns:
        StructuredForwardOutput: The output of the forward pass, including the calculated constraint loss.
    """

    _config_structure: ConfigStructure = {
        "constraint_method": str,
        "data_name": str,
        "loss_coef_latent_constraint": float,
    }

    def __init__(self, cfg: Namespace) -> None:
        super(LatentConstraint, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._calculate_constraint = get_explicit_constraint(
            constraint_name=cfg.constraint_method
        )
        self.constraint_method: str = cfg.constraint_method
        self._data_name: str = cfg.data_name
        self._loss_coef_latent_constraint: float = cfg.loss_coef_latent_constraint

    def forward(self, batch: Batch) -> StructuredForwardOutput:
        loss = self._calculate_constraint(x=batch[self._data_name], dim=None)
        return format_structured_forward_output(
            batch=batch,
            losses=[
                format_structured_loss(
                    loss=loss,
                    coef=self._loss_coef_latent_constraint,
                    name=map_loss_name(loss_name="latent_constraint"),
                    aggregated=True,
                )
            ],
        )


class VectorConditionedLogitsGMPriorNLL(nn.Module):
    r"""
    Module for computing the negative log-likelihood (NLL) loss of a Gaussian Mixture Model (GMM)
    conditioned on a vector of logits.
    Args:
        cfg (Namespace): The configuration object containing the following attributes:
            - data_name (str): The name of the input data tensor.
            - logits_name (str): The name of the logits tensor.
            - components_std (float): The standard deviation of the GMM components.
            - loss_coef_prior_nll (float): The coefficient for scaling the prior NLL loss.
    Attributes:
        _config_structure (ConfigStructure): The structure of the configuration object.
        _data_name (str): The name of the input data tensor.
        logits_name (str): The name of the logits tensor.
        _components_std (float): The standard deviation of the GMM components.
        _loss_coef_prior_nll (float): The coefficient for scaling the prior NLL loss.
    Methods:
        forward(batch: Batch) -> Batch:
            Computes the forward pass of the module.
    """

    _config_structure: ConfigStructure = {
        "data_name": str,
        "logits_name": str,
        "n_components": int,
        "latent_dim": int,
        "components_std": float,
        "loss_coef_prior_nll": float,
    }

    def __init__(self, cfg: Namespace) -> None:
        super(VectorConditionedLogitsGMPriorNLL, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._data_name: str = cfg.data_name
        self._logits_name: str = cfg.logits_name
        self._components_std: float = cfg.components_std
        self._loss_coef_prior_nll: float = cfg.loss_coef_prior_nll
        self._n_components: int = cfg.n_components
        self._latent_dim: int = cfg.latent_dim
        self._component_means = nn.Parameter(
            torch.empty(cfg.n_components, cfg.latent_dim), requires_grad=True
        )
        torch.nn.init.xavier_normal_(self._component_means)
        self.register_buffer(
            "_std", torch.tensor(cfg.components_std, dtype=torch.float32)
        )

    def sample(self, batch: Batch, n_latent_samples: int) -> StructuredForwardOutput:
        logits: torch.Tensor = batch[self._logits_name]
        assert (
            logits.shape[-1] == self._n_components
        ), f"The number of logits is {logits.shape[-1]} but should be {self._n_components}."
        batch_size: int = logits.shape[0]

        # For each condition embedding we are extracting n_latent_samples of latent samples.
        latent_samples_all: torch.Tensor = torch.zeros(
            (batch_size, n_latent_samples, self._latent_dim)
        )
        # Ugh for loop. Write in vectorised way.
        for i in range(batch_size):
            gm_rv = make_gm_rv(
                component_logits=logits[i, :],
                means=self._component_means,
                std=self._std,
            )
            latent_samples: torch.Tensor = gm_rv.sample(
                sample_shape=(n_latent_samples,)
            )
            latent_samples_all[i, :, :] = latent_samples
            #     , pattern="samp dim -> 1 samp dim"
            # )

        # Putting samples to the same place the embedded data is after regular forward.
        batch[self._data_name] = latent_samples_all

        return format_structured_forward_output(batch=batch)

    # def _vectorised_multi_gm_nll(self, logits: torch.Tensor) -> torch.Tensor:
    #     pass

    def forward(self, batch: Batch) -> StructuredForwardOutput:
        logits: torch.Tensor = batch[self._logits_name]
        assert (
            logits.shape[-1] == self._n_components
        ), f"The number of logits is {logits.shape[-1]} but should be {self._n_components}."
        batch_size: int = logits.shape[0]
        x: torch.Tensor = batch[self._data_name]  # (batch_size, n_latent_samples, dim)
        assert (
            len(x.shape) == 3
        ), f"Got len(x.shape) equal {len(x.shape)}, but should be 3."
        assert (
            x.shape[-1] == self._latent_dim
        ), f"The x.shape[-1] should match latent_dim={self._latent_dim}."
        nll_lat_sampl: torch.Tensor = torch.zeros(
            (x.shape[:2])
        )  # (batch_size, n_latent_samples)
        # Ugh for loop. Write in vectorised way.
        for i in range(batch_size):
            gm_rv = make_gm_rv(
                component_logits=logits[i, :],
                means=self._component_means,
                std=self._std,
            )
            nll_lat_sampl[i, :] = -gm_rv.log_prob(x[i, :])

        return format_structured_forward_output(
            batch=batch,
            losses=[
                format_structured_loss(
                    loss=nll_lat_sampl,
                    coef=self._loss_coef_prior_nll,
                    name=map_loss_name(loss_name="prior_nll"),
                    aggregated=False,
                )
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
