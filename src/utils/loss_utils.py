import torch
from torch import Tensor


def kld_stdgaussian(mu: Tensor, logvar: Tensor) -> Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def gex_reconstruction_loss(x_hat: Tensor, x: Tensor) -> Tensor:
    return torch.nn.functional.mse_loss(x_hat, x)


def adt_reconstruction_loss(x_hat: Tensor, x: Tensor) -> Tensor:
    return torch.nn.functional.mse_loss(x_hat, x)
