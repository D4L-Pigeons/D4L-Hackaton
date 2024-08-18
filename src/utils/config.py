from argparse import Namespace
from typing import List

from numpy import isin
from src.utils.common_types import ConfigStructure
import yaml
from src.utils.paths import CONFIG_PATH
import json
from pathlib import Path


def validate_config_structure(
    cfg: Namespace, config_structure: ConfigStructure
) -> None:
    variables = vars(cfg)
    if isinstance(variables, dict):
        if variables.keys() != config_structure.keys():
            raise ValueError(
                f"There is mismatch between provided keys {list(variables.keys())} and expected keys {list(config_structure.keys())}"
            )
        for attr_name, attr_val in variables.items():
            if config_structure.get(attr_name, None) is None:
                raise ValueError(f"Invalid attribute '{attr_name}' in config")
            if isinstance(attr_val, Namespace):
                validate_config_structure(attr_val, config_structure[attr_name])
            elif config_structure[attr_name] is not None and not isinstance(
                attr_val, config_structure[attr_name]
            ):
                raise ValueError(f"Invalid type for attribute '{attr_name}'")
    elif isinstance(variables, list):
        for elem in variables:
            validate_config_structure(elem)


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
