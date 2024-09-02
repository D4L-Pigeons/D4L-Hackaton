import pytest
from torch.optim import Adam, SGD, Adagrad, RMSprop
from torch.optim.lr_scheduler import StepLR, LinearLR, ExponentialLR
from src.models.components.optimizer import (
    get_scheduler,
    get_chained_scheduler,
    ChainedScheduler,
    _SCHEDULERS,
)
from argparse import Namespace
import torch.nn as nn


class FixtureModel(nn.Module):
    def __init__(self):
        super(FixtureModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def params():
    return FixtureModel().parameters()


def test_get_scheduler_valid_name(params):
    optimizer = Adam(params)
    scheduler_name = "step"
    kwargs = {"step_size": 10, "gamma": 0.1}
    scheduler = get_scheduler(scheduler_name, optimizer, kwargs)
    assert isinstance(scheduler, StepLR)


def test_get_scheduler_invalid_name(params):
    optimizer = Adam(params)
    scheduler_name = "invalid_scheduler"
    kwargs = {"step_size": 10, "gamma": 0.1}
    with pytest.raises(ValueError):
        scheduler = get_scheduler(scheduler_name, optimizer, kwargs)


def test_get_scheduler_valid_name_linear(params):
    optimizer = SGD(params)
    scheduler_name = "linear"
    kwargs = {}
    scheduler = get_scheduler(scheduler_name, optimizer, kwargs)
    assert isinstance(scheduler, LinearLR)


def test_get_scheduler_valid_name_exp(params):
    optimizer = Adagrad(params)
    scheduler_name = "exp"
    kwargs = {"gamma": 0.1}
    scheduler = get_scheduler(scheduler_name, optimizer, kwargs)
    assert isinstance(scheduler, ExponentialLR)


def test_get_chained_scheduler(params):
    optimizer = Adam(params)
    schedulers_cfg = [
        Namespace(name="step", kwargs={"step_size": 10, "gamma": 0.1}),
        Namespace(name="linear", kwargs={}),
        Namespace(name="exp", kwargs={"gamma": 0.3}),
    ]
    scheduler = get_chained_scheduler(optimizer, schedulers_cfg)
    assert isinstance(scheduler, ChainedScheduler)
    assert len(scheduler._schedulers) == len(schedulers_cfg)
    for i, scheduler_cfg in enumerate(schedulers_cfg):
        assert isinstance(scheduler._schedulers[i], _SCHEDULERS[scheduler_cfg.name])
        assert scheduler._schedulers[i].optimizer == optimizer
