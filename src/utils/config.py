from argparse import Namespace
from typing import List

from flask import config
from numpy import isin
from src.utils.common_types import ConfigStructure
import yaml
from src.utils.paths import CONFIG_PATH
import json
from pathlib import Path


def validate_config_structure(
    cfg: Namespace | List[Namespace], config_structure: ConfigStructure
) -> None:
    if isinstance(cfg, Namespace):
        variables = vars(cfg)
        if variables.keys() != config_structure.keys():
            raise ValueError(
                f"There is a mismatch between provided keys {list(variables.keys())} and expected keys {list(config_structure.keys())}"
            )
        for attr_name, attr_val in variables.items():
            if (
                isinstance(config_structure[attr_name], dict)
                and isinstance(attr_val, Namespace)
            ) or (
                isinstance(config_structure[attr_name], list)
                and isinstance(attr_val, list)
            ):  # If Namespace we are passing the verification responsibility to some other module.
                validate_config_structure(attr_val, config_structure[attr_name])

            # assert isinstance(
            #     config_structure[attr_name], type
            # ), f"{type(config_structure[attr_name])} {type(attr_val)} {isinstance(config_structure[attr_name], list)} {isinstance(attr_val, list)}"
            elif not isinstance(attr_val, config_structure[attr_name]):
                raise ValueError(
                    f"Invalid type for attribute '{attr_name}'. Expected {config_structure[attr_name]} but got {type(attr_val)}"
                )
    elif isinstance(cfg, list):
        if not isinstance(config_structure, list):
            raise ValueError("Config structure is not list at this level.")
        config_structure = config_structure[0]
        if not config_structure == Namespace:
            for cfg_elem in cfg:
                validate_config_structure(
                    cfg=cfg_elem, config_structure=config_structure
                )


_CHOICE_PATH_SEPARATOR: str = "/"


def parse_choice_spec_path(spec_path: str) -> List[str]:
    return spec_path.split(sep=_CHOICE_PATH_SEPARATOR)


# def load_config(args) -> SimpleNamespace:
#     def load_object(dct):
#         return SimpleNamespace(**dct)

#     with open(CONFIG_PATH / args.method / f"{args.config}.yaml") as file:
#         config_dict = yaml.safe_load(file)
#     config_namespace = json.loads(json.dumps(config_dict), object_hook=load_object)
#     return config_namespace


def _load_config_from_path(file_path: Path) -> Namespace:

    def load_object(dct):
        return Namespace(**dct)

    with open(file_path, "r") as file:
        config_dict = yaml.safe_load(file)
    config_namespace = json.loads(json.dumps(config_dict), object_hook=load_object)
    return config_namespace


def load_config_from_config_dir(file_path: Path) -> Namespace:
    file_path = CONFIG_PATH / file_path
    config = _load_config_from_path(file_path)
    return config

    # # Save config
    # with open(results_path / "config.yaml", "w") as file:
    #     yaml.dump(config.__dict__, file)
    #     print(f"Config saved to: {results_path / 'config.yaml'}")
