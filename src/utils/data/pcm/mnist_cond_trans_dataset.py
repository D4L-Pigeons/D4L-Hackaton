from flask import config
from pkg_resources import get_distribution
import torch
import torch.distributions as td
from typing import List, Dict, Callable, Any, TypedDict, Tuple, TypeAlias
from argparse import Namespace
from src.utils.config import validate_config_structure
from src.utils.common_types import ConfigStructure
import numpy as np
from functools import partial
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from einops import rearrange
import abc
from src.utils.common_types import Batch
from src.utils.paths import RAW_DATA_PATH
import torchvision


_CONTINOUS_DISTRIBUTIONS: Dict[str, td.Distribution] = {"uniform": td.Uniform}


def _get_continous_distribution(dist_name: str, kwargs: Dict) -> td.Distribution:
    distribution = _CONTINOUS_DISTRIBUTIONS.get(dist_name, None)
    if distribution is not None:
        return distribution(
            **{key: torch.tensor(value) for key, value in kwargs.items()}
        )
    raise ValueError(
        f'The provided dist_name {dist_name} is wrong. Must be one of {" ,".join(list(_CONTINOUS_DISTRIBUTIONS.keys()))}'
    )


_CONTINOUS_VALUE_TRANSFORMS: Dict[str, Callable[..., float]] = {
    "identity": lambda x: x,
    "log1p": np.log1p,
}


def _get_continous_value_transform(
    value_transform_name: str, kwargs: Dict
) -> Callable[[float], float]:
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
}


def _get_discrete_distribution(dist_name: str, kwargs: Dict) -> td.Distribution:
    distribution = _DISCRETE_DISTRIBUTIONS.get(dist_name, None)
    if distribution is not None:
        return distribution(
            **{key: torch.tensor(value) for key, value in kwargs.items()}
        )
    raise ValueError(
        f'The provided dist_name {dist_name} is wrong. Must be one of {" ,".join(list(_DISCRETE_DISTRIBUTIONS.keys()))}'
    )


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
        self._size: int = size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        init_size: int = img.shape[1]
        img = transforms.functional.resize(img, size=(self._size, self._size))

        if init_size > self._size:
            img = transforms.functional.pad(
                img,
                padding=[(init_size - self._size) // 2],
            )
        elif init_size < self._size:
            img = transforms.functional.crop(
                img,
                top=(self._size - init_size) // 2,
                left=(self._size - init_size) // 2,
                height=init_size,
                width=init_size,
            )

        img = transforms.functional.resize(
            img,
            size=init_size,
        )
        return img


class TranslateTransform(ImageTransform):
    def __init__(self, translation: Tuple[float, float]) -> None:
        self._translation: Tuple[float, float] = translation

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = transforms.functional.affine(
            img,
            angle=0,
            translate=self._translation,
            scale=1.0,
            shear=0,
        )
        return img


_IMAGE_TENSOR_TRANSFORMS: Dict[str, ImageTransform] = {
    "fade": FadeTransform,
    "rotate": RotationTransform,
    "resize": ResizeTransform,
    "translate": TranslateTransform,
}


def _get_image_transform(transform_name: str, kwargs: Dict[str, Any]) -> ImageTransform:
    transform = _IMAGE_TENSOR_TRANSFORMS.get(transform_name, None)
    if transform is not None:
        return partial(transform, **kwargs)
    raise ValueError(
        f'The provided transform_name {transform_name} is wrong. Must be one of {" ,".join(list(_IMAGE_TENSOR_TRANSFORMS.keys()))}'
    )


class ConditionTransformOutput(TypedDict):
    image_transform: ImageTransform
    condition_value: int


class ConditionTransform(abc.ABC):
    r"""
    Abstract base class for condition transforms.

    Attributes:
        _config_structure (ConfigStructure): The configuration structure for the transform.
        _token_id (int): The token ID.
        _name (str): The name of the transform.
        _argument_keyword (str): The keyword for the image transform.
        _image_transform_constructor (ImageTransform): The image transform constructor.

    Methods:
        __init__(self, cfg: Namespace): Initializes the ConditionTransform instance.
        token_id(self) -> int: Returns the token ID.
        __call__(self) -> ConditionTransformOutput: Abstract method to be implemented by subclasses.
    """

    _config_structure: ConfigStructure = {
        "token_id": int,
        "image_transform": {"name": str, "keyword": str, "kwargs": Namespace},
        "distribution": {"name": str, "kwargs": Namespace},
    }

    def __init__(self, cfg: Namespace) -> None:
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        assert cfg.token_id != 0, "The token_id equal to 0 is reserved for pad token."
        self._token_id: int = cfg.token_id
        self._argument_keyword: str = cfg.image_transform.keyword
        self._image_transform_constructor: ImageTransform = _get_image_transform(
            transform_name=cfg.image_transform.name,
            kwargs=vars(cfg.image_transform.kwargs),
        )

    @property
    def token_id(self) -> int:
        return self._token_id

    @abc.abstractmethod
    def __call__(self) -> ConditionTransformOutput:
        pass


class ContinousConditionTransform(ConditionTransform):
    _config_structure: ConfigStructure = {
        "value_transform": {"name": str, "kwargs": Namespace}
    } | ConditionTransform._config_structure

    def __init__(self, cfg: Namespace) -> None:
        super(ContinousConditionTransform, self).__init__(cfg=cfg)
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._distribution: td.Distribution = _get_continous_distribution(
            dist_name=cfg.distribution.name,
            kwargs=vars(cfg.distribution.kwargs),
        )
        self._cond_value_transform: Callable[[float], float] = (
            _get_continous_value_transform(
                value_transform_name=cfg.value_transform.name,
                kwargs=vars(cfg.value_transform.kwargs),
            )
        )

    def __call__(self) -> ConditionTransformOutput:
        value = self._distribution.sample(sample_shape=(1,)).item()
        value = self._cond_value_transform(value)
        kwargs = {self._argument_keyword: value}
        output = ConditionTransformOutput(
            image_transform=self._image_transform_constructor(**kwargs),
            condition_value=torch.tensor(value, dtype=torch.float32),
        )
        return output


class CategoricalConditionTransform(ConditionTransform):
    _config_structure: ConfigStructure = {
        "categories_map": (int | float, [float | int]),
    } | ConditionTransform._config_structure

    def __init__(self, cfg: Namespace) -> None:
        super(CategoricalConditionTransform, self).__init__(cfg=cfg)
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._distribution: td.Distribution = _get_discrete_distribution(
            dist_name=cfg.distribution.name, kwargs=vars(cfg.distribution.kwargs)
        )
        self._categories: List[float | List[float]] = cfg.categories_map

    def _map_category_to_value(self, category: int) -> int:
        return self._categories[category]

    def __call__(self) -> ConditionTransformOutput:
        category = self._distribution.sample(sample_shape=(1,)).item()
        value = self._map_category_to_value(category=category)
        output = ConditionTransformOutput(
            image_transform=self._image_transform_constructor(
                **{self._argument_keyword: value}
            ),
            condition_value=torch.tensor(
                category, dtype=torch.float32
            ),  # Category is of float type to allow categorical and coninous conditions values in one sequence tensor.
        )
        return output


class ConditionTransformManagerOutput(TypedDict):
    transform: transforms.Compose
    token_ids: torch.Tensor
    values: torch.Tensor


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
        self._cond_transforms: List[ConditionTransform] = []
        self._n_conditions: int = len(cfg)
        for i, cond_cfg in enumerate(cfg):
            assert (
                cond_cfg.known_prob >= 0 and cond_cfg.known_prob <= 1
            ), f"The value of known_prob attribute is {cond_cfg.known_prob} but needs to be within [0, 1] interval."
            self._cond_known_probs[i] = cond_cfg.known_prob

            self._cond_transforms.append(
                ContinousConditionTransform(cfg=cond_cfg.cfg)
                if cond_cfg.is_continous
                else CategoricalConditionTransform(cfg=cond_cfg.cfg)
            )

    def __call__(self) -> ConditionTransformManagerOutput:
        image_transforms: List[ImageTransform] = []
        # Default the token_ids to pad token and values to zero.
        token_ids: torch.Tensor = torch.zeros(self._n_conditions, dtype=torch.long)
        values: torch.Tensor = torch.zeros(self._n_conditions, dtype=torch.float32)

        known_mask = torch.bernoulli(self._cond_known_probs)
        i: int = 0

        for is_known, cond_transform in zip(known_mask, self._cond_transforms):
            output = cond_transform()
            image_transforms.append(output["image_transform"])
            # Update the token_id and value only for non-pad conditions. Consequently the padding is added at the end.
            if is_known:
                token_ids[i] = cond_transform.token_id
                values[i] = output["condition_value"]
                i += 1

        return ConditionTransformManagerOutput(
            transform=transforms.Compose(transforms=image_transforms),
            token_ids=token_ids,
            values=values,
        )


class ConditionalMNIST(Dataset):
    r"""
    A dataset class for conditional MNIST data.

    Args:
        cfg (Namespace): The configuration object containing the dataset parameters.

    Attributes:
        _label_token_id (torch.Tensor): The token ID for the label.
        _label_known_prob (float): The probability of the label being known.
        _mnist (torchvision.datasets.MNIST): The MNIST dataset.
        _cond_trans_mgr (ConditionTransformManager): The condition transformation manager.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(index): Returns the item at the given index.

    """

    _config_structure: ConfigStructure = {
        "train": bool,
        "label_condition": {"token_id": int, "known_prob": float},
        "cond_trans_mgr": [Namespace],
    }

    def __init__(self, cfg: Namespace) -> None:
        super(ConditionalMNIST, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        assert (
            cfg.label_condition.token_id != 0
        ), "The token_id equal to 0 is reserved for pad token."
        self._label_token_id: torch.Tensor = torch.tensor(
            [cfg.label_condition.token_id], dtype=torch.long
        )

        self._label_known_prob: float = cfg.label_condition.known_prob

        self._mnist = torchvision.datasets.MNIST(
            root=RAW_DATA_PATH, train=cfg.train, download=True
        )

        self._cond_trans_mgr: ConditionTransformManager = ConditionTransformManager(
            cfg=cfg.cond_trans_mgr
        )

    def __len__(self) -> int:
        return len(self._mnist)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        conditions: ConditionTransformManagerOutput = self._cond_trans_mgr()

        img, label = self._mnist.__getitem__(index=index)
        label = torch.tensor([label], dtype=torch.float32)
        img = transforms.ToTensor()(img)
        img = conditions["transform"](img)
        token_ids: torch.Tensor = conditions["token_ids"]
        values: torch.Tensor = conditions["values"]

        if torch.bernoulli(torch.tensor(self._label_known_prob)).item():
            token_ids = torch.cat([self._label_token_id, token_ids], dim=0)
            values = torch.cat([label, values], dim=0)
        else:
            token_ids = torch.cat(
                [token_ids, torch.zeros((1,), dtype=torch.long)], dim=0
            )
            values = torch.cat([values, torch.zeros((1,))], dim=0)

        return {
            "img": img,
            "condition_token_ids": token_ids,
            "condition_values": values,
        }


def cmnist_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Batch:
    # Collect the images, condition token IDs, and condition values from the batch
    images = [item["img"] for item in batch]
    condition_token_ids = [item["condition_token_ids"] for item in batch]
    condition_values = [item["condition_values"] for item in batch]
    # Stack the condition token IDs and values into tensors
    images = torch.stack(images, dim=0)
    condition_token_ids = torch.stack(condition_token_ids, dim=0)
    condition_values = torch.stack(condition_values, dim=0)

    return {
        "img": images,
        "condition_token_ids": condition_token_ids,
        "condition_values": condition_values,
    }


def get_ConditionalMnistDataloader(
    cmnist: Dataset, batch_size: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        dataset=cmnist,
        collate_fn=cmnist_collate_fn,
        batch_size=batch_size,
        shuffle=shuffle,
    )
