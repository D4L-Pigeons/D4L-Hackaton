from types import SimpleNamespace

import pytest
import torch
import torch.distributions as td
import torch.nn.functional as F
from numpy import require

from src.models.components.latent import (
    GaussianPosterior,
    _format_forward_output,
    _get_std_transform,
    _sample_diff_gm_rv,
    _sample_nondiff,
    make_gm_rv,
    make_normal_rv,
)


def test_format_forward_output():
    batch = {"data": torch.tensor([1, 2, 3])}
    losses = [{"loss": torch.tensor(0.5)}]
    output = _format_forward_output(batch, losses)
    assert output["batch"] == batch
    assert output["losses"] == losses


def test_make_normal_rv():
    mean = torch.tensor([0.0])
    std = torch.tensor([1.0])
    rv = make_normal_rv(mean, std)
    assert isinstance(rv, td.Normal)
    assert torch.equal(rv.loc, mean)
    assert torch.equal(rv.scale, std)


def test_make_gm_rv():
    component_logits = torch.tensor([0.0, 1.0])
    means = torch.tensor([[0.0], [1.0]])
    std = 1.0
    rv = make_gm_rv(component_logits, means, std)
    assert isinstance(
        rv, td.MixtureSameFamily
    ), "rv is not an instance of td.MixtureSameFamily"
    assert (
        rv.component_distribution.base_dist.loc.shape == means.shape
    ), "The shape of means and component_distribution.loc are not equal."
    assert (
        rv.component_distribution.base_dist.scale.shape == means.shape
    ), "The shape of means and component_distribution.scale are not equal."
    # For some reason the logits are shifted when passed to Categorical and this needs to be taken into account when constructing a test.
    diff = rv.mixture_distribution.logits - component_logits
    single_entry = diff[0]
    assert torch.equal(
        diff, torch.ones_like(diff) * single_entry
    ), f"mixture_distribution logits are not equal to component_logits. They are off by {torch.sum(torch.abs(rv.mixture_distribution.logits - component_logits))}"

    assert torch.allclose(
        rv.component_distribution.base_dist.loc, means
    ), "The means of the component distribution are not equal."


def test_sample_nondiff():
    rv = td.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
    sample_shape = torch.Size([2])
    samples = _sample_nondiff(rv, sample_shape)
    assert samples.shape == torch.Size([2, 1])

    # Additional test for Gaussian Mixture
    component_logits = torch.tensor([0.0, 1.0], requires_grad=True)
    means = torch.tensor([[0.0], [1.0]], requires_grad=True)
    std = 1.0
    gm_rv = make_gm_rv(component_logits, means, std)
    gm_samples = _sample_nondiff(gm_rv, sample_shape)

    assert gm_samples.shape == torch.Size([2, 1])
    assert gm_samples.sum().requires_grad == False, "Samples require_grad==True"


def test_sample_diff_gm_rv_gradient():
    component_logits = torch.tensor([0.0, 1.0], requires_grad=True)
    means = torch.tensor([[0.0], [1.0]], requires_grad=True)
    std = 1.0
    rv = make_gm_rv(component_logits, means, std)
    sample_shape = torch.Size([2])
    samples = _sample_diff_gm_rv(rv, sample_shape)
    assert samples.shape == torch.Size([2, 1])

    # Check if the gradient flows correctly
    assert (
        component_logits.grad is None
    ), "Gradient for component_logits is not None before backward"
    assert means.grad is None, "Gradient for means is not None before backward"
    samples.sum().backward()
    assert component_logits.grad is not None, "Gradient for component_logits is None"
    assert means.grad is not None, "Gradient for means is None"


def test_get_std_transform():
    transform = _get_std_transform("softplus")
    x = torch.tensor([-1.0, 0.0, 1.0])
    transformed_x = transform(x)
    assert torch.all(transformed_x > 0)


def test_gaussian_posterior():
    cfg = SimpleNamespace(std_transformation="softplus", data_name="data")
    model = GaussianPosterior(cfg)
    assert model._std_transformation == _get_std_transform("softplus")
    assert model._data_name == "data"


if __name__ == "__main__":
    pytest.main()
