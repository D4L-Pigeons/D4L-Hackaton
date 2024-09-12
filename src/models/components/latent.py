r"""
This module contains components for working with latent variables in a machine learning model, specifically focusing on Gaussian and Gaussian Mixture Models (GMMs). It includes functions and classes for creating random variables, computing negative log-likelihoods, sampling, and performing forward passes in a neural network.

Functions:
    - make_normal_rv(mean: torch.Tensor, std: torch.Tensor) -> GaussianRV:
    - make_gm_rv(component_logits: torch.Tensor, means: torch.Tensor, std: float) -> GaussianMixtureRV:
    - vectorized_nll_gmm(logits: torch.Tensor, means: torch.Tensor, std: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
    - _sample_nondiff(rv: td.distribution.Distribution, sample_shape: torch.Size | Tuple[int]) -> torch.Tensor:
    - _sample_diff_gm_rv(gm_rv: GaussianMixtureRV, sample_shape: torch.Size) -> torch.Tensor:
    - calc_component_logits_for_gm_sample(rv: GaussianMixtureRV, x: torch.Tensor) -> torch.Tensor:
        Calculate the component logits for a Gaussian Mixture sample.
    - _get_std_transform(transform_name: str) -> Callable[[torch.Tensor], torch.Tensor]:

Classes:
    - GaussianPosterior(nn.Module):
        Represents a Gaussian posterior module, inheriting from nn.Module. It is used to model the posterior distribution in a variational autoencoder setup.
    - _GM(nn.Module, abc.ABC):
    - GaussianMixturePriorNLL(_GM):
        Implements a Gaussian Mixture model with a prior negative log-likelihood loss.
    - FuzzyClustering(_GM):
        Implements fuzzy clustering in the latent space using Gaussian Mixture models.
    - LatentConstraint(nn.Module):
        LatentConstraint is a PyTorch module that applies a specified constraint method to the latent space of a model.
    - VectorConditionedLogitsGMPriorNLL(nn.Module):
        A PyTorch module that represents a Gaussian Mixture Prior with Negative Log-Likelihood (NLL) loss, conditioned on vector logits.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from einops import rearrange, einsum
from argparse import Namespace
from typing import Callable, Dict, NamedTuple, Optional, Tuple, TypeAlias, Any
import abc

from utils.common_types import (
    Batch,
    StructuredForwardOutput,
    format_structured_forward_output,
    format_structured_loss,
    ConfigStructure,
)
from models.components.loss import get_explicit_constraint, map_loss_name
from utils.config import validate_config_structure


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


# its raw gpt generated proposition. TO BE TESTED AND COMPATED WITH NONVECTORISED VERSION - this one is actually pure chat just not to forget about vectorization
def vectorized_nll_gmm(
    logits: torch.Tensor, means: torch.Tensor, std: torch.Tensor, samples: torch.Tensor
) -> torch.Tensor:
    r"""
    Compute the negative log-likelihood of a Gaussian Mixture Model in a vectorized way.

    Args:
        logits (torch.Tensor): Logits for the mixture components, shape (batch_size, n_components).
        means (torch.Tensor): Means of the mixture components, shape (n_components, n_features).
        std (torch.Tensor): Standard deviations of the mixture components, shape (n_components, n_features).
        samples (torch.Tensor): Samples for which to compute the NLL, shape (batch_size, n_samples, n_features).

    Returns:
        torch.Tensor: The negative log-likelihood for each sample in the batch, shape (batch_size,).
    """
    batch_size, n_samples, n_features = samples.shape
    n_components = logits.shape[1]

    # Expand dimensions to match shapes for broadcasting
    logits = logits.unsqueeze(1)  # Shape: (batch_size, 1, n_components)
    means = means.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, n_components, n_features)
    std = std.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, n_components, n_features)
    samples = samples.unsqueeze(2)  # Shape: (batch_size, n_samples, 1, n_features)

    # Create the normal distributions for each component
    normal_dists = td.Normal(means, std)

    # Compute the log probabilities for each component
    log_probs = normal_dists.log_prob(
        samples
    )  # Shape: (batch_size, n_samples, n_components, n_features)
    log_probs = log_probs.sum(
        dim=-1
    )  # Sum over the feature dimension, shape: (batch_size, n_samples, n_components)

    # Add the logits to the log probabilities
    log_probs = log_probs + logits  # Shape: (batch_size, n_samples, n_components)

    # Compute the log-sum-exp over the components
    log_sum_exp = torch.logsumexp(log_probs, dim=-1)  # Shape: (batch_size, n_samples)

    # Compute the negative log-likelihood
    nll = -log_sum_exp.mean(dim=-1)  # Shape: (batch_size,)

    return nll


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

    # Repeate logits to create priors for each sample.
    pattern = f'comps -> {" ".join(["1"] * len(sample_shape))} comps'
    repeated_logits = rearrange(
        tensor=gm_rv.mixture_distribution.logits, pattern=pattern
    ).repeat(*sample_shape, 1)

    # Use gumbel trick to select component for each sample.
    sampled_components_one_hots: torch.Tensor = F.gumbel_softmax(
        repeated_logits,
        dim=1,
        hard=True,
    )  # (*sample_shape, n_components)

    # Sample from each component for each sample.
    component_samples: torch.Tensor = gm_rv.component_distribution.rsample(
        sample_shape=sample_shape
    )  # (*sample_shape, n_components, dim)

    # Select the sampled component for each sample.
    selected_component_samples = (
        sampled_components_one_hots.unsqueeze(-1) * component_samples
    ).sum(
        dim=-2
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
    r"""
    Returns the standard transform function based on the provided transform_name.

    Parameters:
        transform_name (str): The name of the transform.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: The standard transform function.

    Raises:
        ValueError: If the provided transform_name is not one of the available transform names.
    """

    std_transform = _STD_TRANSFORMS.get(transform_name, None)
    if std_transform is not None:
        return std_transform
    raise ValueError(
        f'The provided transform_name {transform_name} is wrong. Must be one of {" ,".join(list(_STD_TRANSFORMS.keys()))}'
    )


class GaussianPosterior(nn.Module):
    r"""
    This class represents a Gaussian posterior module, inheriting from nn.Module. It is used to model the posterior distribution in a variational autoencoder setup.

    Parameters:
    - cfg (Namespace): The configuration for the nn.Module.

    Methods:
    - forward(batch: Batch, data_name: str, n_latent_samples: int) -> StructuredForwardOutput:
        Performs the forward pass of the module.

    Static Methods:
    - _parse_hparams_to_dict(cfg: Namespace) -> Dict[str, Any]:
        Parses the hyperparameters to a dictionary for logging purposes.

    Attributes:
    - _cfg_structure (ConfigStructure): The configuration structure for the module.
    - _std_transform (Callable): The function to transform standard deviations.
    - _loss_coef_posterior_entropy (float): The coefficient for the posterior entropy loss.

    """

    _cfg_structure: ConfigStructure = {
        "std_transform": str,
        "loss_coef_posterior_entropy": float,
    }

    def __init__(self, cfg: Namespace) -> None:
        super(GaussianPosterior, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._cfg_structure)

        # Initialize standard deviation transformation.
        self._std_transform: Callable = _get_std_transform(cfg.std_transform)

        # Initialize loss base loss coefficient, which might be rescaled with scheduler.
        self._loss_coef_posterior_entropy: float = cfg.loss_coef_posterior_entropy

    @staticmethod
    def _parse_hparams_to_dict(cfg: Namespace) -> Dict[str, Any]:
        r"""
        Parses the hyperparameters from the given Namespace object and returns them as a dictionary used for logging.

        Args:
            cfg (Namespace): The Namespace object containing the hyperparameters.

        Returns:
            Dict[str, Any]: A dictionary containing the parsed hyperparameters.

        """

        return {
            "std_transform": cfg.std_transform,
            "loss_coef_posterior_entropy": cfg.loss_coef_posterior_entropy,
        }

    def forward(
        self, batch: Batch, data_name: str, n_latent_samples: int
    ) -> StructuredForwardOutput:
        r"""
           Forward pass of the Latent component. Input data shape: (batch dim). Output data shape: (batch sampl dim//2)

        Args:
            batch (Batch): Input batch data.
            data_name (str): Name of the data in the batch.
            n_latent_samples (int): Number of latent samples.

        Returns:
            StructuredForwardOutput: Output of the forward pass.

        """

        # Split the output of the encoder into two vectors of the same size representing multivariate gaussian mean and sqrt of diagonal covariance matrix.
        means, stds = batch[data_name].chunk(chunks=2, dim=1)

        # Apply transformation of the standard deviations.
        stds = self._std_transform(stds)

        # Create random variable form pytorch.distributions.
        rv = make_normal_rv(mean=means, std=stds)

        # Calculate aggregated entropy going from 1-dim gaussian to multivariate gaussian.
        neg_entropy_per_batch_sample = -einsum(
            rv.entropy(), "batch dim -> batch"
        ).unsqueeze(
            1
        )  # (batch_size, 1)

        # Replacing the 'data' with reparametrisation trick samples in the batch and leaving the rest unchanged.
        latent_samples = rv.rsample(
            sample_shape=(n_latent_samples,)
        )  # (n_latent_samples, batch_size, latent_dim)

        # Rearranging the dimensions for further processing.
        batch[data_name] = rearrange(
            tensor=latent_samples, pattern="samp batch dim -> batch samp dim"
        )

        return format_structured_forward_output(
            batch=batch,
            losses=[
                format_structured_loss(
                    loss=neg_entropy_per_batch_sample,
                    coef=self._loss_coef_posterior_entropy,
                    name=map_loss_name(loss_name="posterior_neg_entropy"),
                    reduced=False,
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
        "components_std": float,
        "latent_dim": int,
    }

    def __init__(self, cfg: Namespace) -> None:
        super(_GM, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        # Initialize Gaussian Mixture parameters i.e. logits, means and std for isotropic Gaussian Mixtures.
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

        # Initialize placeholder for torch.distribution distribution.
        self._rv: Optional[td.MixtureSameFamily[td.Normal]] = None

    @staticmethod
    def _parse_hparams_to_dict(cfg: Namespace) -> Dict[str, Any]:
        r"""
        Parses hyperparameters from a Namespace object to a dictionary.

        Args:
            cfg (Namespace): A Namespace object containing hyperparameters.

        Returns:
            Dict[str, Any]: A dictionary with keys 'n_components', 'components_std',
                            and 'latent_dim' mapped to their respective values from cfg.
        """

        return {
            "n_components": cfg.n_components,
            "components_std": cfg.components_std,
            "latent_dim": cfg.latent_dim,
        }

    def set_rv(self) -> None:
        r"""
        Sets the random variable (rv) for the model component.

        This method initializes the `_rv` attribute using the `make_gm_rv` function.
        The random variable is created based on the component logits, means, and standard deviation.

        Args:
            None

        Returns:
            None
        """

        self._rv = make_gm_rv(
            component_logits=self._component_logits,
            means=self._component_means,
            std=self._std,
        )

    def reset_rv(self) -> None:
        r"""
        Resets the random variable (_rv) to None.

        This method sets the internal attribute `_rv` to None, effectively
        resetting any previously stored random variable.
        """

        self._rv = None

    def sample(self, n_samples: int) -> Batch:
        r"""
        Samples from the random variable set in the instance.

        Args:
            n_samples (int): The number of samples to generate.

        Returns:
            Batch: A batch of samples formatted as a structured forward output.

        Raises:
            RuntimeError: If the random variable is not set before sampling.
        """

        if self._rv is None:
            raise RuntimeError(
                "Random variable not set. Call set_rv() before sampling."
            )

        # Sample differentiably from the Gaussian Mixture from each component.
        component_samples = _sample_diff_gm_rv(self._rv, sample_shape=(n_samples,))

        # Create a batch with only the samples for teh
        batch = {
            self._data_name: rearrange(
                tensor=component_samples, pattern="samp batch dim -> batch samp dim"
            )
        }

        return format_structured_forward_output(batch=batch)

    def _get_nll(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the negative log-likelihood (NLL) of the given tensor `x`.

        Args:
            x (torch.Tensor): The input tensor for which the NLL is to be computed.

        Returns:
            torch.Tensor: The computed NLL for each latent sample.
        """

        gm_nll_per_lat_sampl = -self._rv.log_prob(x)

        return gm_nll_per_lat_sampl

    def _get_component_conditioned_nll(
        self, x: torch.Tensor, component_indicator: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Computes the negative log-likelihood (NLL) of the given data `x` conditioned on the specified component indicators.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_latent_samples, dim).
            component_indicator (torch.Tensor): Tensor indicating the component for each sample in the batch, of shape (batch_size,).

        Returns:
            torch.Tensor: The NLL of the data `x` for each latent sample, of shape (batch_size, n_latent_samples).
        """

        # Rearrange the tensor to calculate per component negative log likelihood.
        x = rearrange(x, "batch sampl dim -> batch sampl 1 dim")

        # Compute nll per Gaussian Mixture component.
        gm_nll_per_component = -self._rv.component_distribution.log_prob(
            x
        )  # (batch_size, n_latent_samples, n_components)

        # Choose the likelihood of the specified component.
        gm_nll_per_lat_sampl = gm_nll_per_component[
            torch.arange(component_indicator.shape[0], dtype=torch.long),
            :,
            component_indicator.to(torch.long),
        ]  # (batch_size, n_latent_samples)

        return gm_nll_per_lat_sampl

    @abc.abstractmethod
    def forward(self, batch: Batch) -> StructuredForwardOutput:
        pass


class GaussianMixturePriorNLL(_GM):
    _config_structure: ConfigStructure = {
        "loss_coef_prior_nll": float,
    } | _GM._config_structure  # Adding the config structure of the parent class.

    def __init__(self, cfg: Namespace) -> None:
        super(GaussianMixturePriorNLL, self).__init__(cfg=cfg)
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        # Initialize loss coefficient which may be rescaled with loss coefficient scheduler.
        self._loss_coef_prior_nll: float = cfg.loss_coef_prior_nll

    @staticmethod
    def _parse_hparams_to_dict(cfg: Namespace) -> Dict[str, Any]:
        r"""
        Parses hyperparameters from a Namespace object to a dictionary.

        Args:
            cfg (Namespace): A Namespace object containing hyperparameters.

        Returns:
            Dict[str, Any]: A dictionary containing the parsed hyperparameters,
                            including an additional key "loss_coef_prior_nll"
                            from the cfg object.
        """

        return _GM._parse_hparams_to_dict(cfg=cfg) | {
            "loss_coef_prior_nll": cfg.loss_coef_prior_nll
        }

    def forward_unknown(self, batch: Batch, data_name: str) -> StructuredForwardOutput:
        r"""
        Perform a forward pass for unknown data.
        Args:
            batch (Batch): The input batch containing data.
            data_name (str): The name of the data within the batch to process.
        Returns:
            StructuredForwardOutput: The structured output containing the loss information.
        Raises:
            AssertionError: If the shapes of `nll_lat_sampl` and `unknown_gm_nll_lat_sampl` do not match.
        """

        x = batch[data_name]

        self.set_rv()

        nll_lat_sampl = torch.zeros((x.shape[:2]))  # (batch_size, n_latent_samples)

        unknown_gm_nll_lat_sampl = self._get_nll(x=x)  # (batch_size, n_latent_samples)

        assert (
            nll_lat_sampl.shape == unknown_gm_nll_lat_sampl.shape
        ), f"nll_lat_sampl.shape = {nll_lat_sampl.shape} and unknown_gm_nll_lat_sampl.shape = {unknown_gm_nll_lat_sampl.shape}"

        self.reset_rv()

        return format_structured_forward_output(
            batch=batch,
            losses=[
                format_structured_loss(
                    loss=unknown_gm_nll_lat_sampl,
                    coef=self._loss_coef_prior_nll,
                    name=map_loss_name(loss_name="prior_nll"),
                    reduced=False,
                )
            ],
        )

    def forward_known(
        self, batch: Batch, data_name: str, component_indicator_name: str
    ) -> StructuredForwardOutput:
        r"""
        Perform a forward pass using known data and component indicators.

        Args:
            batch (Batch): The input batch containing data and component indicators.
            data_name (str): The key to access the data in the batch.
            component_indicator_name (str): The key to access the component indicators in the batch.

        Returns:
            StructuredForwardOutput: The output containing the structured forward results.
        """

        x = batch[data_name]
        component_indicator = batch[component_indicator_name]

        known_gm_nll_lat_sampl = self._get_component_conditioned_nll(
            x=x,
            component_indicator=component_indicator,
        )  # (known_batch_size, n_latent_samples)

        self.reset_rv()

        return format_structured_forward_output(
            batch=batch,
            losses=[
                format_structured_loss(
                    loss=known_gm_nll_lat_sampl,
                    coef=self._loss_coef_prior_nll,
                    name=map_loss_name(loss_name="prior_nll"),
                    reduced=False,
                )
            ],
        )

    def forward_mixed(
        self, batch: Batch, data_name: str, component_indicator_name: str
    ) -> StructuredForwardOutput:
        r"""
        Perform a forward pass with mixed known and unknown components.

        Args:
            batch (Batch): The input batch containing data and component indicators.
            data_name (str): The key to access the data in the batch.
            component_indicator_name (str): The key to access the component indicators in the batch.

        Returns:
            StructuredForwardOutput: The output containing the negative log-likelihood losses.
        """

        x = batch[data_name]

        component_indicator = batch[component_indicator_name]
        unknown_mask = (
            component_indicator == -1
        )  # -1 indicates that the component is unknown for a sample
        known_mask = unknown_mask.logical_not()

        self.set_rv()

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
                    reduced=False,
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

    Static Methods:
        _parse_hparams_to_dict(cfg: Namespace) -> Dict[str, Any]: Parses hyperparameters from a Namespace object to a dictionary.
    """

    _config_structure: ConfigStructure = {
        "constraint_method": str,
        "loss_coef_latent_fuzzy_clustering": float,
        "loss_coef_clustering_component_reg": float,
    } | _GM._config_structure

    def __init__(self, cfg: Namespace) -> None:
        super(FuzzyClustering, self).__init__(cfg=cfg)
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        # Setup function calculating explicit latent constraint like L2 on component means bringing them to the center of the latent space.
        self._calculate_constraint = get_explicit_constraint(
            constraint_name=cfg.constraint_method
        )

        # Initialize loss coefficients for fuzzy clustering loss aka Gaussian Mixture nll and component regularization.
        self._loss_coef_latent_fuzzy_clustering: float = (
            cfg.loss_coef_latent_fuzzy_clustering
        )
        self._loss_coef_clustering_component_reg: float = (
            cfg.loss_coef_clustering_component_reg
        )

    @staticmethod
    def _parse_hparams_to_dict(cfg: Namespace) -> Dict[str, Any]:
        r"""
        Parses hyperparameters from a Namespace object to a dictionary.

        Args:
            cfg (Namespace): A Namespace object containing hyperparameters.

        Returns:
            Dict[str, Any]: A dictionary containing the parsed hyperparameters,
                            including additional keys for constraint method and
                            loss coefficients related to latent fuzzy clustering
                            and clustering component regularization.
        """

        return _GM._parse_hparams_to_dict(cfg=cfg) | {
            "constraint_method": cfg.constraint_method,
            "loss_coef_latent_fuzzy_clustering": cfg.loss_coef_latent_fuzzy_clustering,
            "loss_coef_clustering_component_reg": cfg.loss_coef_clustering_component_reg,
        }

    def _calculate_component_constraint(self) -> torch.Tensor:
        r"""
        Calculates the component constraint for the model.

        This method computes the regularization term for the component means,
        which is weighted by the components' probabilities. The regularization
        helps in bringing the components' means closer together.

        Returns:
            torch.Tensor: The computed component regularization term.
        """

        # Component mean regularization bringing the components' means. Weighted by the components' probabilities.
        component_regularization = (
            self._calculate_constraint(x=self._component_means, dim=1)
            * torch.softmax(self._component_logits.detach(), dim=0)
        ).sum()  # (1)

        return component_regularization

    def forward(self, batch: Batch, data_name: str) -> StructuredForwardOutput:
        r"""
        Perform a forward pass through the model.

        Args:
            batch (Batch): The input batch of data.
            data_name (str): The name of the data field in the batch.

        Returns:
            StructuredForwardOutput: The output of the forward pass, including losses.

        This method performs the following steps:
        1. Sets the random variables (RV) for the model.
        2. Computes the negative log-likelihood (NLL) for the Gaussian Mixture (GM) model.
        3. Calculates the component regularization constraint.
        4. Resets the random variables (RV).
        5. Formats and returns the structured forward output, including the GM NLL loss and the component regularization loss.
        """

        self.set_rv()

        gm_nll = self._get_nll(x=batch[data_name])  # (batch, sampl)
        comp_reg = self._calculate_component_constraint()

        self.reset_rv()

        return format_structured_forward_output(
            batch=batch,
            losses=[
                format_structured_loss(
                    loss=gm_nll,
                    coef=self._loss_coef_latent_fuzzy_clustering,
                    name=map_loss_name(loss_name="latent_fuzzy_clustering"),
                    reduced=False,
                ),
                format_structured_loss(
                    loss=comp_reg,
                    coef=self._loss_coef_clustering_component_reg,
                    name=map_loss_name(loss_name="clustering_component_reg"),
                    reduced=True,
                ),
            ],
        )


class LatentConstraint(nn.Module):
    r"""
    LatentConstraint is a PyTorch module that applies a specified constraint method to the latent space of a model.

    Attributes:
        _config_structure (ConfigStructure): A dictionary defining the expected configuration structure.
        constraint_method (str): The name of the constraint method to be applied.
        _loss_coef_latent_constraint (float): The coefficient for the latent constraint loss.

    Methods:
        __init__(cfg: Namespace) -> None:
            Initializes the LatentConstraint module with the given configuration.

        forward(batch: Batch) -> StructuredForwardOutput:
            Applies the constraint method to the specified data in the batch and returns the structured forward output.

    Static Methods:
        _parse_hparams_to_dict(cfg: Namespace) -> Dict[str, Any]:
            Converts the configuration parameters to a dictionary.
    """

    _config_structure: ConfigStructure = {
        "constraint_method": str,
        "loss_coef_latent_constraint": float,
    }

    def __init__(self, cfg: Namespace) -> None:
        super(LatentConstraint, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        # Set up explicit constraint on samples provided in the batch.
        self._calculate_constraint = get_explicit_constraint(
            constraint_name=cfg.constraint_method
        )

        # Initialize constrain method and loss coefficient, which might be scaled with scheduler.
        self.constraint_method: str = cfg.constraint_method
        self._loss_coef_latent_constraint: float = cfg.loss_coef_latent_constraint

    @staticmethod
    def _parse_hparams_to_dict(cfg: Namespace) -> Dict[str, Any]:
        r"""
        Parses hyperparameters from a Namespace object to a dictionary.

        Args:
            cfg (Namespace): A Namespace object containing hyperparameters.

        Returns:
            Dict[str, Any]: A dictionary with the parsed hyperparameters.
        """

        return {
            "constraint_method": cfg.constraint_method,
            "loss_coef_latent_constraint": cfg.loss_coef_latent_constraint,
        }

    def forward(self, batch: Batch, data_name: str) -> StructuredForwardOutput:
        r"""
        Perform a forward pass to calculate the constraint loss and format the output.

        Args:
            batch (Batch): The input batch containing data.
            data_name (str): The key to access specific data within the batch.

        Returns:
            StructuredForwardOutput: The formatted output containing the calculated loss.
        """

        loss = self._calculate_constraint(x=batch[data_name], dim=None)

        return format_structured_forward_output(
            batch=batch,
            losses=[
                format_structured_loss(
                    loss=loss,
                    coef=self._loss_coef_latent_constraint,
                    name=map_loss_name(loss_name="latent_constraint"),
                    reduced=True,
                )
            ],
        )


class VectorConditionedLogitsGMPriorNLL(nn.Module):
    r"""
    A PyTorch module that represents a Gaussian Mixture Prior with Negative Log-Likelihood (NLL) loss,
    conditioned on vector logits.

    Attributes:
        _config_structure (ConfigStructure): The expected configuration structure.
        _components_std (float): Standard deviation of the components.
        _loss_coef_prior_nll (float): Coefficient for the prior NLL loss.
        _n_components (int): Number of components in the Gaussian Mixture.
        _latent_dim (int): Dimensionality of the latent space.
        _component_means (nn.Parameter): Means of the Gaussian components.
        _std (torch.Tensor): Standard deviation tensor for the components.

    Methods:
        __init__(cfg: Namespace) -> None:
            Initializes the module with the given configuration.

        get_means(batch: Batch, gm_means_name: str) -> StructuredForwardOutput:
            Retrieves the means of the Gaussian components and adds them to the batch.

        sample(batch: Batch, n_latent_samples: int, data_name: str, logits_name: str) -> StructuredForwardOutput:
            Samples latent variables from the Gaussian Mixture based on the provided logits.

        forward(batch: Batch, data_name: str, logits_name: str) -> StructuredForwardOutput:
            Computes the forward pass, calculating the negative log-likelihood of the latent samples.

    Static Methods:
        _parse_hparams_to_dict(cfg: Namespace) -> Dict[str, Any]:
            Parses hyperparameters from the configuration to a dictionary.
    """

    _config_structure: ConfigStructure = {
        "n_components": int,
        "latent_dim": int,
        "components_std": float,
        "loss_coef_prior_nll": float,
    }

    def __init__(self, cfg: Namespace) -> None:
        super(VectorConditionedLogitsGMPriorNLL, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        # Setup Gaussian Mixture parameters independent of conditions i.e. component means and buffered standard deviations.
        self._component_means = nn.Parameter(
            torch.empty(cfg.n_components, cfg.latent_dim), requires_grad=True
        )
        torch.nn.init.xavier_normal_(self._component_means)
        self.register_buffer(
            "_std", torch.tensor(cfg.components_std, dtype=torch.float32)
        )

        # Initialize loss coefficients, which might be altered with the scheduler.
        self._components_std: float = cfg.components_std
        self._loss_coef_prior_nll: float = cfg.loss_coef_prior_nll

        # Save hparams for logging.
        self._n_components: int = cfg.n_components
        self._latent_dim: int = cfg.latent_dim

    @staticmethod
    def _parse_hparams_to_dict(cfg: Namespace) -> Dict[str, Any]:
        r"""
        Parses hyperparameters from a Namespace object to a dictionary.

        Args:
            cfg (Namespace): A Namespace object containing the hyperparameters.

        Returns:
            Dict[str, Any]: A dictionary containing the parsed hyperparameters.
        """

        return {
            "n_components": cfg.n_components,
            "latent_dim": cfg.latent_dim,
            "components_std": cfg.components_std,
            "loss_coef_prior_nll": cfg.loss_coef_prior_nll,
        }

    def get_means(self, batch: Batch, gm_means_name: str) -> StructuredForwardOutput:
        r"""
        Updates the batch with component means and returns a structured forward output.

        Args:
            batch (Batch): The input batch to be updated.
            gm_means_name (str): The key under which the component means will be stored in the batch.

        Returns:
            StructuredForwardOutput: The structured forward output containing the updated batch.
        """

        batch[gm_means_name] = self._component_means

        return format_structured_forward_output(batch=batch)

    def sample(
        self, batch: Batch, n_latent_samples: int, data_name: str, logits_name: str
    ) -> StructuredForwardOutput:
        r"""
        Samples latent variables from a Gaussian mixture model for each item in the batch.

        Args:
            batch (Batch): The input batch containing data and logits.
            n_latent_samples (int): The number of latent samples to generate for each item in the batch.
            data_name (str): The key in the batch where the latent samples will be stored.
            logits_name (str): The key in the batch where the logits are stored.

        Returns:
            StructuredForwardOutput: The batch with the latent samples added under the specified key.

        Raises:
            AssertionError: If the number of provided logits does not match the expected number of components.
        """

        logits: torch.Tensor = batch[logits_name]
        assert (
            logits.shape[-1] == self._n_components
        ), f"The number of provided logits is {logits.shape[-1]} but should be {self._n_components}."

        batch_size: int = logits.shape[0]

        # For each condition embedding we are extracting n_latent_samples of latent samples.
        latent_samples_all: torch.Tensor = torch.zeros(
            (batch_size, n_latent_samples, self._latent_dim)
        )

        # Ugh for loop. Write in vectorised way -> See gpt generated prooposition at the top of the file.
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

        # Putting samples to the same place the embedded data is after regular forward.
        batch[data_name] = latent_samples_all

        return format_structured_forward_output(batch=batch)

    def _vectorised_multi_gm_nll(self, logits: torch.Tensor) -> torch.Tensor:
        pass

    def forward(
        self, batch: Batch, data_name: str, logits_name: str
    ) -> StructuredForwardOutput:
        r"""
        Perform the forward pass for the latent model component.
        Args:
            batch (Batch): The input batch containing data and logits.
            data_name (str): The key to access the data tensor in the batch.
            logits_name (str): The key to access the logits tensor in the batch.
        Returns:
            StructuredForwardOutput: The output containing the structured loss.
        Raises:
            AssertionError: If the shape of logits or data tensor does not match the expected dimensions.
        Notes:
            - The logits tensor should have the last dimension equal to the number of components.
            - The data tensor should have three dimensions with the last dimension equal to the latent dimension.
            - The function computes the negative log-likelihood for each latent sample using a Gaussian mixture model.
        """

        logits: torch.Tensor = batch[logits_name]

        assert (
            logits.shape[-1] == self._n_components
        ), f"The number of provided logits is {logits.shape[-1]} but should be {self._n_components}."

        batch_size: int = logits.shape[0]
        x: torch.Tensor = batch[data_name]  # (batch_size, n_latent_samples, dim)

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
                    reduced=False,
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
