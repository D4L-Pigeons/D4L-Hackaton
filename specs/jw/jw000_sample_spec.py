from mrunner.helpers.specification_helper import create_experiments_helper

from src.utils.parser import parse_args, combine_with_defaults

name = globals()["script"][:-3]

base_config = {
    "train": True,
}
base_config = combine_with_defaults(base_config, defaults=vars(parse_args([])))

params_grid = []

print(params_grid)


experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="",
    script="python3 -u mrun.py",
    python_path="",
    exclude=[
        "apptainer",
        "venv",
    ],
    tags=[name],
    base_config=base_config,
    params_grid=params_grid,
)

# REMEMBER THAT YOU NEED TO EXPORT NEPTUNE_API_TOKEN, NEPTUNE_PROJECT AND SET PYTHONPATH. Example:
# export NEPTUNE_API_TOKEN="your_token"
# export NEPTUNE_PROJECT="multimodal/vaes"
# export PYTHONPATH=$PYTHONPATH:$(pwd)
