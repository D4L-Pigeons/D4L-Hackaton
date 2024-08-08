from mrunner.helpers.specification_helper import create_experiments_helper

from src.global_utils.parser import combine_with_defaults, parse_args

name = globals()["script"][:-3]

base_config = {
    "model_name": "OmiAE",
    "mode": "train",
    "hidden_dim": 128,
    "encoder_hidden_dim": 128,
    "encoder_out_dim": 16,
    "latent_dim": 16,
    "decoder_hidden_dim": 128,
}
base_config = combine_with_defaults(base_config, modalities_names=["gex", "adt"])

params_grid = []
params_grid += [
    {
        "method": ["omivae"],
    }
]

print(params_grid)


experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="multimodal/vaes",
    script="python3 -u mrun.py",
    python_path="",
    exclude=["examples", "old_code", "notebooks", "data"],
    tags=[name],
    base_config=base_config,
    params_grid=params_grid,
)
