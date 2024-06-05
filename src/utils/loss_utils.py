from typing import Tuple

import torch
import torch.optim as optim
from torch import Tensor
from tqdm import tqdm


def kld_stdgaussian(mu: Tensor, logvar: Tensor) -> Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def gex_reconstruction_loss(x_hat: Tensor, x: Tensor) -> Tensor:
    return torch.nn.functional.mse_loss(x_hat, x)


def adt_reconstruction_loss(x_hat: Tensor, x: Tensor) -> Tensor:
    return torch.nn.functional.mse_loss(x_hat, x)


def estimate_initial_negative_binomial_params(counts: Tensor) -> Tuple[Tensor]:
    """
    Estimate the initial dispersion parameter (alpha) for each gene using a moment-based approach.

    Parameters:
    counts (Tensor): The count matrix with genes as rows and cells as columns.

    Returns:
    Tuple[Tensor]: A tuple of initial mean counts and dispersions for each gene.
    """
    mean_counts = counts.mean(dim=1)
    var_counts = counts.var(dim=1)
    dispersions = (var_counts - mean_counts) / (mean_counts**2)
    dispersions = dispersions.clamp(min=1e-10)  # Ensure non-negative dispersions
    return mean_counts, dispersions


def negative_binomial_log_likelihood(
    x: Tensor, mu: Tensor, theta: Tensor, eps: float = 1e-10
) -> Tensor:
    r"""
    Compute the negative binomial log likelihood for the given data.

    Arguments:
    x : torch.Tensor
        The data.
    mu : torch.Tensor
        The mean of the data.
    theta : torch.Tensor
        The dispersion parameter.

    Returns:
    nll : torch.Tensor
        The negative binomial log likelihood.
    """
    assert eps > 0, f"eps must be positive, got {eps} instead."

    return (
        -theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
        - x * (torch.log(theta + eps) - torch.log(theta + mu + eps))
        - torch.lgamma(x + theta + eps).sum()
        + torch.lgamma(x + 1).sum()
        + torch.lgamma(theta + eps).sum()
    )


def estimate_negative_binomial_parameters(
    counts: Tensor, eps: float = 1e-10
) -> Tuple[Tensor]:
    """
    Estimate the mean and dispersion parameters for each gene using the negative binomial log likelihood.

    Parameters:
    counts (Tensor): The count matrix with cells as rows and featuers as columns.
    mu (Tensor): The mean of the data.
    theta (Tensor): The initial dispersion parameter.
    eps (float): A small value to ensure numerical stability.

    Returns:
    Tuple[Tensor]: A tuple of mean counts and dispersions for each gene.
    """
    means, dispersions = estimate_initial_negative_binomial_params(counts)
    n_features = counts.shape[1]
    progress_bar = tqdm(range(n_features), desc="Estimating NB parameters")
    for i in progress_bar:
        mean = means[i].clone()
        mean.requires_grad_(True)
        dispersion = dispersions[i].clone()
        dispersion.requires_grad_(True)
        optimizer = optim.Adam([mean, dispersion], lr=0.01)
        for _ in range(100):
            optimizer.zero_grad()
            nll = negative_binomial_log_likelihood(counts[:, i], mean, dispersion, eps)
            nll.backward()
            optimizer.step()
        means[i] = mean.item()
        dispersions[i] = dispersion.item()
        progress_bar.set_postfix_str(
            f"Feature {i + 1}/{n_features} - NLL: {nll.item():.2f}"
        )
