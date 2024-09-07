from argparse import Namespace
from typing import List
import yaml
from utils.paths import CONFIG_PATH
import json
from pathlib import Path

from utils.common_types import ConfigStructure


def validate_config_structure(
    cfg: Namespace | List[Namespace] | None, config_structure: ConfigStructure
) -> None:
    r"""
    Validates the structure of the configuration object `cfg` against the expected `config_structure`.

    Args:
        cfg (Namespace | List[Namespace] | None): The configuration object to be validated.
            It can be a single Namespace object, a list of Namespace objects, or None.
        config_structure (ConfigStructure): The expected structure of the configuration object.

    Raises:
        ValueError: If the structure of the configuration object does not match the expected structure.

    Notes:
        - Tuple allows to drop a config element or allows success when _validate_config_structure returns None on any of the possibilities from the tuple.
        - The `Namespace` type represents a simple object that holds attributes as key-value pairs.
        - The `List[Namespace]` type represents a list of Namespace objects.
        - The `None` type represents an empty configuration object.

    The function recursively checks the structure of the `cfg` object against the `config_structure`.
    It performs the following checks based on the type of `config_structure`:
    - If `config_structure` is a type, it checks if `cfg` is an instance of that type.
    - If `config_structure` is a tuple, it checks if `cfg` matches any of the elements in the tuple.
    - If `config_structure` is a Namespace object, it checks if the keys of `cfg` match the keys of `config_structure`.
      It then recursively checks the structure of each attribute in `cfg`.
    - If `config_structure` is a list, it checks if `cfg` is a list and recursively checks the structure of each element in the list.

    If any mismatch or error is found during the validation process, a ValueError is raised with a descriptive error message.
    """

    def _validate_config_structure(
        cfg: Namespace | List[Namespace] | None,
        config_structure: ConfigStructure | None,
        cfg_path: str = "",
    ) -> None | Exception:
        error = None

        if cfg is None and config_structure is None:
            return error

        if isinstance(config_structure, type):
            if not isinstance(cfg, config_structure):
                error = ValueError(
                    f"Invalid type for attribute '{cfg_path}' with value '{cfg}'. Expected {config_structure} but got {type(cfg)}"
                )

        elif isinstance(config_structure, tuple):
            if cfg is not None:
                for config_structure_ in config_structure:
                    error = _validate_config_structure(cfg, config_structure_, cfg_path)
                    if error is None:
                        break

        elif isinstance(cfg, Namespace):
            cfg = vars(cfg)

            if cfg.keys() != config_structure.keys():
                error = ValueError(
                    f"At level {cfg_path} | There is a mismatch between provided keys {list(cfg.keys())} and expected keys {list(config_structure.keys())}"
                )

            for attr_name, attr_val in cfg.items():
                error = _validate_config_structure(
                    cfg=attr_val,
                    config_structure=config_structure[attr_name],
                    cfg_path=cfg_path + "/" + attr_name,
                )
                if error is not None:
                    break

        elif isinstance(cfg, list):
            if not isinstance(config_structure, list):
                error = ValueError(
                    f"At level {cfg_path} | Config structure is of type {type(config_structure)} not a list as expected."
                )
            elif (
                len(cfg) != 0
            ):  # Proceed only if the list is nonempty. BAD if the list is obligatory!!!
                config_structure = config_structure[0]
                if not config_structure == Namespace:
                    for cfg_elem in cfg:
                        _validate_config_structure(
                            cfg=cfg_elem,
                            config_structure=config_structure,
                            cfg_path=cfg_path + "[list-item]/",
                        )

        return error

    error = _validate_config_structure(cfg=cfg, config_structure=config_structure)

    if error is not None:
        raise error


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


def load_config_from_path(file_path: Path) -> Namespace:

    def load_object(dct):
        return Namespace(**dct)

    with open(file_path, "r") as file:
        config_dict = yaml.safe_load(file)
    config_namespace = json.loads(json.dumps(config_dict), object_hook=load_object)
    return config_namespace


def load_config_from_config_dir(file_path: Path) -> Namespace:
    file_path = CONFIG_PATH / file_path
    config = load_config_from_path(file_path)
    return config

    # # Save config
    # with open(results_path / "config.yaml", "w") as file:
    #     yaml.dump(config.__dict__, file)
    #     print(f"Config saved to: {results_path / 'config.yaml'}")
