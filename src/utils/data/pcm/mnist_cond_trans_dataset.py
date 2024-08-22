from flask import config
from pkg_resources import get_distribution
from sympy import Q
import torch
import torch.distributions as td
from typing import List, Dict, Callable, Any, TypedDict, Typle, TypeAlias
from argparse import Namespace
from src.utils.config import validate_config_structure
from src.utils.common_types import ConfigStructure
import numpy as np
from functools import partial
import PIL as pil
from torch.utils.data import Dataset
from torchvision import transforms
from einops import rearrange
import abc
from src.utils.common_types import Batch

_CONTINOUS_DISTRIBUTIONS: Dict[td.Distribution] = {
    "uniform": td.Uniform,
}


def _get_continous_distribution(dist_name: str, kwargs: Dict) -> td.Distribution:
    distribution = _CONTINOUS_DISTRIBUTIONS.get(dist_name, None)
    if distribution is not None:
        return distribution(**kwargs)
    raise ValueError(
        f'The provided dist_name {dist_name} is wrong. Must be one of {" ,".join(list(_CONTINOUS_DISTRIBUTIONS.keys()))}'
    )


def continous_value_transforms_wrapper(
    func: Callable[..., Any]
) -> Callable[[Dict[str, Any]], Callable[..., Any]]:
    def wrapper(kwargs: Dict[str, Any]) -> Callable[..., Any]:
        return partial(func, **kwargs)

    return wrapper


ContinousValueTransform: TypeAlias = Callable[[float], float]

_CONTINOUS_VALUE_TRANSFORMS: Dict[str, ContinousValueTransform] = {
    "identity": continous_value_transforms_wrapper(func=lambda x: x),
    "log1p": continous_value_transforms_wrapper(func=np.log1p),
}


def _get_continous_value_transform(
    value_transform_name: str, kwargs: Dict
) -> ContinousValueTransform:
    value_transform = _CONTINOUS_VALUE_TRANSFORMS.get(value_transform_name, None)
    if value_transform is not None:
        return partial(value_transform, **kwargs)
    raise ValueError(
        f'The provided value_transform_name {value_transform_name} is wrong. Must be one of {" ,".join(list(_CONTINOUS_VALUE_TRANSFORMS.keys()))}'
    )


_DISCRETE_DISTRIBUTIONS: Dict[str, td.Distribution] = {
    "bernoulli": td.Bernoulli,
    "categorical": td.Categorical,
    "binomial": td.Binomial,
    # Add other discrete distributions as needed
}


def _get_discrete_distribution(dist_name: str, kwargs: Dict) -> td.Distribution:
    distribution = _DISCRETE_DISTRIBUTIONS.get(dist_name, None)
    if distribution is not None:
        return distribution(**kwargs)
    raise ValueError(
        f'The provided dist_name {dist_name} is wrong. Must be one of {" ,".join(list(_DISCRETE_DISTRIBUTIONS.keys()))}'
    )


CategoriesRemapper: TypeAlias = Callable[[int], int]


def _get_categories_remapper(kwargs: Dict[str, Any]) -> CategoriesRemapper:
    return partial(func=lambda x, values: values[x], **kwargs)


class ImageTransform(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self):
        pass


class FadeTransform(ImageTransform):
    def __init__(self, threshold: float) -> None:
        self._threshold: float = threshold
        assert (
            threshold >= 0 and threshold < 1
        ), "The threshold is not within [0, 1) interval."

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = (
            torch.clamp(img - self._threshold, min=0, max=1) * 1 / (1 - self._threshold)
        )
        return img


class RotationTransform(ImageTransform):
    def __init__(self, degree: float) -> None:
        self._rotation: transforms.RandomRotation = transforms.RandomRotation(
            degrees=[degree, degree]
        )

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self._rotation(img)


class ResizeTransform(ImageTransform):
    def __init__(self, size: int) -> None:
        self._size = size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = transforms.functional.resize(
            transforms.functional.pad(
                transforms.functional.resize(img, size=(self._size, self._size)),
                padding=[(img.shape[0] - self._size) // 2],
            ),
            size=img.shape,
        )
        return img


_IMAGE_TENSOR_TRANSFORMS: Dict[str, ImageTransform] = {
    "fade": FadeTransform,
    "rotate": RotationTransform,
    "resize": ResizeTransform,
}


def _get_image_transform(transform_name: str, kwargs: Dict[str, Any]) -> ImageTransform:
    transform = _IMAGE_TENSOR_TRANSFORMS.get(transform_name, None)
    if transform is not None:
        return partial(transform, **kwargs)
    raise ValueError(
        f'The provided transform_name {transform_name} is wrong. Must be one of {" ,".join(list(_IMAGE_TENSOR_TRANSFORMS.keys()))}'
    )


class Condition(TypedDict):
    condition_name: str
    data: torch.Tensor


class ConditionTransformOutput(TypedDict):
    image_transform: ImageTransform
    condition: Condition


class ConditionTransform(abc.ABC):
    _config_structure: ConfigStructure = {
        "data_name": str,
        "image_transform": {"name": str, "keyword": str, "kwargs": Namespace},
        "distribution": {"name": str, "kwargs": Namespace},
    }

    def __init__(self, cfg: Namespace) -> None:
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._data_name: str = cfg.data_name
        self._argument_keyword: str = cfg.image_transform.keyword
        self._image_transform_constructor: ImageTransform = _get_image_transform(
            transform_name=cfg.transform.name
        )

    @abc.abstractmethod
    def __call__(self) -> ConditionTransformOutput:
        pass


class ContinousConditionTransform(ConditionTransform):
    _config_structure: ConfigStructure = {
        {
            "value_transform": {"name": str, "kwargs": Namespace},
        }
        | ConditionTransform._config_structure
    }

    def __init__(self, cfg: Namespace) -> None:
        super(ContinousConditionTransform, self).__init__(cfg=cfg)
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._distribution: td.Distribution = _get_continous_distribution(
            dist_name=cfg.distribution.name,
            kwargs=vars(cfg.distribution.kwargs),
        )
        self._cond_value_transform: ContinousValueTransform = (
            _get_continous_value_transform(
                value_transform_name=cfg.value_transform.name,
                kwargs=vars(cfg.value_transform.kwargs),
            )
        )

    def __call__(self) -> ConditionTransformOutput:
        value = self._distribution.sample((1)).item()
        value = self._cond_value_transform(value)
        output = ConditionTransformOutput(
            image_transform=self._image_transform_constructor(
                **{self._argument_keyword: value}
            ),
            condition=Condition(
                condition_name=self._data_name,
                data=torch.tensor(value, dtype=torch.float32),
            ),
        )
        return output


class CategoricalConditionTransform(ConditionTransform):
    _config_structure: ConfigStructure = {
        {
            "categories_map": [int],
        }
        | ConditionTransform._config_structure
    }

    def __init__(self, cfg: Namespace) -> None:
        super(ContinousConditionTransform, self).__init__(cfg=cfg)
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._distribution: td.Distribution = _get_discrete_distribution(
            dist_name=cfg.distribution, kwargs=vars(cfg.distibution.kwargs)
        )
        self._categories: List[int] = cfg.categories_map

    def __call__(self) -> ConditionTransformOutput:
        category = self._distribution.sample((1)).item()
        value = self._categories_map[value]
        output = ConditionTransformOutput(
            image_transform=self._image_transform_constructor(
                **{self._argument_keyword: value}
            ),
            condition=Condition(
                condition_name=self._data_name,
                data=torch.tensor(category, dtype=torch.int),
            ),
        )
        return output


class ConditionTransformManagerOutput(TypedDict):
    transform: transforms.Compose
    conditions: Dict[str, torch.Tensor]


class ConditionTransformManager:
    _config_structure: ConfigStructure = [
        {
            "known_prob": float,
            "is_continous": bool,
            "cfg": Namespace,
        }
    ]

    def __init__(self, cfg: Namespace) -> None:
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._cond_known_probs: torch.Tensor = torch.zeros(len(cfg))
        self._cond_transforms: ConditionTransform = []
        for i, cond_cfg in enumerate(cfg):
            assert (
                cond_cfg.known_prob >= 0 and cond_cfg.known_prob <= 1
            ), f"The value of known_prob attribute is {cond_cfg.known_prob} but needs to be within [0, 1] interval."
            self._cond_known_probs[i] = cond_cfg.known_prob

            self._cond_transforms.append(
                ContinousConditionTransform(cfg=cond_cfg.cfg)
                if cond_cfg.is_continous
                else CategoricalConditionTransform(cfg=cond_cfg)
            )

    def get_transform(self) -> None:
        image_transforms: List[ImageTransform] = []
        conditions: Batch = {}

        for cond_transform in self._cond_transforms:
            output = cond_transform()
            image_transforms.append(output["image_transform"])
            conditions[output["condition"]["data_name"]] = output["condition"]["value"]

        return ConditionTransformManagerOutput(
            transform=transforms.Compose(transforms=image_transforms),
            conditions=conditions,
        )


from src.utils.paths import RAW_DATA_PATH
import torchvision


class ConditionalMNIST(Dataset):
    _config_structure: ConfigStructure = {"train": bool, "cond_trans_mgr": Namespace}

    def __init__(self, cfg: Namespace) -> None:
        super(ConditionalMNIST, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._mnist = torchvision.datasets.MNIST(
            root=RAW_DATA_PATH, train=cfg.train, download=True
        )

        self._cond_trans_mgr: ConditionTransformManager = ConditionTransformManager(
            cfg=cfg.cond_trans_mgr
        )
