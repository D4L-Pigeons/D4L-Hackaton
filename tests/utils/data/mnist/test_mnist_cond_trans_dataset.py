from json import load
import pytest
import torch
import torch.distributions as td
from argparse import Namespace
from src.utils.data.pcm.mnist_cond_trans_dataset import (
    ConditionTransform,
    ContinousConditionTransform,
    FadeTransform,
    CategoricalConditionTransform,
    ConditionTransformManager,
)
from src.utils.config import load_config_from_path
from pathlib import Path
import pytest
from argparse import Namespace


class ConditionTransformSubclass(ConditionTransform):
    def __call__(self) -> None:
        return None


def test_ConditionTransform_init():
    cfg = load_config_from_path(
        file_path=Path("D4L-Hackaton")
        / "tests"
        / "utils"
        / "data"
        / "mnist"
        / "dummy_cfgs"
        / "dummy_cfg-ConditionTransform.yaml"
    )
    ct = ConditionTransformSubclass(cfg=cfg)


def test_ContinousConditionTransform_init():
    cfg = load_config_from_path(
        file_path=Path("D4L-Hackaton")
        / "tests"
        / "utils"
        / "data"
        / "mnist"
        / "dummy_cfgs"
        / "dummy_cfg-ContinousConditionTransform.yaml"
    )
    cct = ContinousConditionTransform(cfg=cfg)
    assert isinstance(cct._distribution, td.Uniform)


def test_ContinousConditionTransform_call():
    cfg = load_config_from_path(
        file_path=Path("D4L-Hackaton")
        / "tests"
        / "utils"
        / "data"
        / "mnist"
        / "dummy_cfgs"
        / "dummy_cfg-ContinousConditionTransform.yaml"
    )
    cct = ContinousConditionTransform(cfg=cfg)
    output = cct()
    assert isinstance(output["image_transform"], FadeTransform)
    assert output["condition_value"] <= 1 and output["condition_value"] >= 0


def test_CategoricalConditionTransform_init():
    cfg = load_config_from_path(
        file_path=Path("D4L-Hackaton")
        / "tests"
        / "utils"
        / "data"
        / "mnist"
        / "dummy_cfgs"
        / "dummy_cfg-CategoricalConditionTransform.yaml"
    )
    cct = CategoricalConditionTransform(cfg=cfg)
    assert isinstance(cct._distribution, td.Categorical)


def test_CategoricalConditionTransform_call():
    cfg = load_config_from_path(
        file_path=Path("D4L-Hackaton")
        / "tests"
        / "utils"
        / "data"
        / "mnist"
        / "dummy_cfgs"
        / "dummy_cfg-CategoricalConditionTransform.yaml"
    )
    cct = CategoricalConditionTransform(cfg=cfg)
    output = cct()
    assert isinstance(output["image_transform"], FadeTransform)
    assert isinstance(output["condition_value"], torch.Tensor)


def test_ConditionTransformManager_get_transform1():
    cfg = load_config_from_path(
        file_path=Path("D4L-Hackaton")
        / "tests"
        / "utils"
        / "data"
        / "mnist"
        / "dummy_cfgs"
        / "dummy_cfg-ConditionTransformManager1.yaml"
    )
    cond_trans_mgr = ConditionTransformManager(cfg=cfg)
    output = cond_trans_mgr.get_transform()

    assert len(output["transform"].transforms) == 2
    assert len(output["token_ids"].tolist()) > 0
    assert len(output["values"].tolist()) > 0


def test_ConditionTransformManager_get_transform2():
    cfg = load_config_from_path(
        file_path=Path("D4L-Hackaton")
        / "tests"
        / "utils"
        / "data"
        / "mnist"
        / "dummy_cfgs"
        / "dummy_cfg-ConditionTransformManager2.yaml"
    )
    cond_trans_mgr = ConditionTransformManager(cfg=cfg)
    output = cond_trans_mgr.get_transform()

    assert len(output["transform"].transforms) == 2
    assert len(output["token_ids"].tolist()) == 2
    assert len(output["values"].tolist()) == 2
    assert output["token_ids"].tolist() == [0, 0]
    assert output["values"].tolist() == [0, 0]
