from types import SimpleNamespace
from argparse import Namespace
import pytest
import torch
import torch.distributions as td
import torch.nn.functional as F
from numpy import require
import torch
import pytest
from types import SimpleNamespace
from src.models.components.latent import GaussianPosterior
import torch
import pytest
from src.models.components.latent import _GM
import torch
import pytest
from src.models.components.latent import _GM
import torch
import pytest
from src.models.components.latent import _GM, GaussianMixtureRV, make_gm_rv
import torch
import pytest
from types import SimpleNamespace
from src.models.components.latent import GMPriorNLL
import torch
import pytest
from src.models.components.latent import calc_mmd
import torch
from src.models.components.latent import calc_pairwise_distances
import torch
import pytest
from src.models.components.latent import calc_mmd
import pytest
import torch
from argparse import Namespace
from src.models.components.latent import LatentConstraint
import pytest
import torch
from argparse import Namespace
from src.models.components.latent import FuzzyClustering

from src.models.components.latent import (
    GaussianPosterior,
    _format_forward_output,
    _get_std_transform,
    _sample_diff_gm_rv,
    _sample_nondiff,
    make_gm_rv,
    make_normal_rv,
    GMPriorNLL,
    FuzzyClustering,
    calc_component_logits_for_gm_sample,
    calc_pairwise_distances,
    calc_mmd,
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
    dim = 8
    means = torch.cat([torch.ones(1, dim), torch.zeros(1, dim)], dim=0)
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


def test_calc_component_logits_for_gm_sample():
    component_neg_log_probs = torch.tensor([0.5, 0.5]).log()
    means = torch.tensor([[0.0], [1000.0]])
    std = 1.0
    rv = make_gm_rv(component_neg_log_probs, means, std)
    x = torch.tensor([[0], [1000.0]])
    logits = calc_component_logits_for_gm_sample(rv, x)
    assert logits[0][0] > logits[0][1]
    assert logits[1][0] < logits[1][1]


def test_get_std_transform():
    transform = _get_std_transform("softplus")
    x = torch.tensor([-1.0, 0.0, 1.0])
    transformed_x = transform(x)
    assert torch.all(transformed_x > 0)


def test_gaussian_posterior():
    cfg = SimpleNamespace(
        n_latent_samples=2, std_transformation="softplus", data_name="data"
    )
    model = GaussianPosterior(cfg)

    # Test forward method
    batch = {"data": torch.tensor([[1.0, 2.0, 3.0]])}
    output = model.forward(batch)
    assert "batch" in output
    assert "losses" in output

    # Test _std_transformation attribute
    assert model._std_transformation is not None

    # Test _data_name attribute
    assert model._data_name == "data"


def test_gaussian_posterior_forward():
    cfg = SimpleNamespace(
        n_latent_samples=2, std_transformation="softplus", data_name="data"
    )
    model = GaussianPosterior(cfg)

    # Test forward method
    batch = {"data": torch.tensor([[1.0, 2.0, 3.0, 3.0]])}
    data_init_shape = batch["data"].shape
    output = model.forward(batch)
    assert "batch" in output
    assert "losses" in output
    assert (
        data_init_shape[0] == output["batch"]["data"].shape[0]
    ), "The batch shape do not match after the forward."
    assert (
        2 == output["batch"]["data"].shape[1]
    ), "The number of latent samples is not correct in the output."
    assert (
        data_init_shape[1] == 2 * output["batch"]["data"].shape[3]
    ), "The dimension after forward is not a half of the inputted."
    assert (
        output["batch"]["data"].shape[2] == 1
    ), "The dummy dimension was not properly added or the dimensions are mixed somehow."


class _GM_subclass(_GM):
    def __init__(self, cfg: Namespace) -> None:
        super(_GM_subclass, self).__init__(cfg=cfg)

    def forward(self) -> None:
        return None


def test_GM_set_reset_rv():
    cfg = SimpleNamespace(
        n_components=2, latent_dim=3, components_std=1.0, data_name="data"
    )
    model = _GM_subclass(cfg)
    assert model._rv is None
    model.set_rv()
    assert isinstance(model._rv, GaussianMixtureRV)
    model.reset_rv()
    assert model._rv is None


@pytest.fixture
def _GM_sample_tensor():
    batch_size, n_latent_samples, dim = 3, 5, 8
    tensor = torch.arange(
        batch_size * n_latent_samples * dim, dtype=torch.float32
    ).reshape(batch_size, n_latent_samples, dim)
    return tensor


def test_GM_get_nll(_GM_sample_tensor):
    cfg = SimpleNamespace(
        n_components=2, latent_dim=8, components_std=1.0, data_name="data"
    )
    model = _GM_subclass(cfg)
    model.set_rv()
    x = _GM_sample_tensor
    nll = model._get_nll(x)
    assert (nll >= 0).all()
    assert nll.shape == x.shape[:-1]


def test_GM_get_component_conditioned_nll(_GM_sample_tensor):
    cfg = SimpleNamespace(
        n_components=2, latent_dim=8, components_std=1.0, data_name="data"
    )
    model = _GM_subclass(cfg)
    model.set_rv()
    x = _GM_sample_tensor
    component_indicator = torch.tensor([0, 1, 0])
    nll_lat_sampl = model._get_component_conditioned_nll(x, component_indicator)
    assert (nll_lat_sampl >= 0).all()
    assert nll_lat_sampl.shape == x.shape[:-1]


@pytest.fixture
def batch_fixture():
    batch_size, n_latent_samples, dim = 3, 5, 8
    tensor = torch.arange(
        batch_size * n_latent_samples * dim, dtype=torch.float32
    ).reshape(batch_size, n_latent_samples, dim)
    return {
        "data": tensor,
        "component_indicator": torch.tensor([-1, 0, 1]),
    }


def test_GMPriorNLL_forward(batch_fixture):
    cfg = SimpleNamespace(n_components=2, latent_dim=8, components_std=1.0)
    model = GMPriorNLL(
        cfg, data_name="data", component_indicator_name="component_indicator"
    )
    # Test forward method
    batch = batch_fixture
    output = model.forward(batch)
    assert "batch" in output
    assert "losses" in output
    assert output["batch"] == batch
    assert len(output["losses"]) == 1
    assert output["losses"][0]["name"] == "prior_nll"
    assert output["losses"][0]["data"].shape == batch["data"].shape[:2]


def test_calc_mmd():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    kernel_name = "rbf"
    latent_dim = 3
    components_std = 1.0
    mmd = calc_mmd(x, y, kernel_name, latent_dim, components_std)
    assert isinstance(mmd, torch.Tensor)
    assert mmd.shape == torch.Size([])
    assert mmd >= 0.0


def test_calc_pairwise_distances_positive():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    distances = calc_pairwise_distances(x, y)
    assert (distances.dist_xx >= 0).all()
    assert (distances.dist_xy >= 0).all()
    assert (distances.dist_yy >= 0).all()


def test_calc_mmd():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    kernel_name = "rbf"
    latent_dim = 3
    components_std = 1.0

    mmd = calc_mmd(x, y, kernel_name, latent_dim, components_std)

    assert isinstance(mmd, torch.Tensor)
    assert mmd.shape == torch.Size([])
    assert mmd.item() >= 0.0

    # Additional test case
    x = torch.tensor([[1.0, 2.0, 3.0]])
    y = torch.tensor([[4.0, 5.0, 6.0]])
    kernel_name = "inverse_multiquadratic"
    latent_dim = 3
    components_std = 1.0

    mmd = calc_mmd(x, y, kernel_name, latent_dim, components_std)

    assert isinstance(mmd, torch.Tensor)
    assert mmd.shape == torch.Size([])
    assert mmd.item() >= 0.0


def test_latent_constraint():
    cfg = Namespace(constraint_method="l2")
    data_name = "latent_data"
    model = LatentConstraint(cfg, data_name)

    # Test forward method
    batch = {"latent_data": torch.tensor([1.0, 2.0, 3.0])}
    output = model.forward(batch)
    assert "batch" in output
    assert "losses" in output

    # Test _calculate_constraint attribute
    assert model._calculate_constraint is not None

    # Test _data_name attribute
    assert model._data_name == "latent_data"

    assert (output["losses"][0]["data"] >= 0).all()


def test_FuzzyClustering_forward():
    cfg = Namespace(
        n_components=2,
        latent_dim=3,
        components_std=1.0,
        constraint_method="huber",
    )
    model = FuzzyClustering(cfg, data_name="data")

    # Test _calculate_component_constraint method
    component_reg = model._calculate_component_constraint()
    assert isinstance(component_reg, torch.Tensor)

    # Test forward method
    batch = {"data": torch.tensor([[1.0, 2.0, 3.0]])}
    output = model.forward(batch)
    assert "batch" in output
    assert "losses" in output
    losses = output["losses"]
    assert len(losses) == 2
    assert "fuzz_clust" == losses[0]["name"]
    assert "comp_clust_reg" == losses[1]["name"]
    assert isinstance(losses[0]["data"], torch.Tensor)
    assert isinstance(losses[1]["data"], torch.Tensor)


if __name__ == "__main__":
    pytest.main()
