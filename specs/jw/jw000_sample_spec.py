from mrunner.helpers.specification_helper import create_experiments_helper

from src.global_utils.parser import combine_with_defaults, parse_args

name = globals()["script"][:-3]

base_config = {
    "model_name": "OmiAE",
    "mode": "train",
    "path_to_dataset": "/home/asia/data/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad",
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
        # "train_samples": [0.08],
    }
]

print(params_grid)


experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="multimodal/vaes",
    script="python3 -u mrun.py",
    python_path="",
    exclude=["examples", "old_code", "notes", "checkpoints", "notebooks", "data"],
    tags=[name],
    base_config=base_config,
    params_grid=params_grid,
)


# from mrunner.helpers.specification_helper import create_experiments_helper

# from src.global_utils.parser import combine_with_defaults, parse_args

# name = globals()["script"][:-3]


# base_config = {
#     "method": "omivae",
#     # "model_name": "OmiAE",
#     # "mode": "train",
#     # "gex_dim": 13953,  # max 13953 | MAY BE CHANGED FOR DEBUGGING
#     # "adt_dim": 134,  # max 134 | MAY BE CHANGED FOR DEBUGGING

#     # "path_to_dataset": "/data/raw/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad",
#     # "retrain": True,
#     # "lr": 0.001,
#     # "batch_size": 32,
#     # "subsample_frac": 0.1,  # MAY BE CHANGED FOR DEBUGGING
#     # "data_normalization": "standarize",  # "log1p", "standardize", "pearson_residuals", null -> None
#     # "remove_batch_effect": True,
#     # # include_class_labels: False
#     # "target_hierarchy_level": -1,
#     # "max_epochs": 10,
#     # "log_every_n_steps": 1,
#     # "early_stopping": True,
#     # "early_stopping_delta": 0.001,
#     # "early_stopping_patience": 5,
#     # "hidden_dim": 128,
#     # "encoder_hidden_dim": 128,
#     # "encoder_out_dim": 16,
#     # "latent_dim": 16,
#     # "decoder_hidden_dim": 128,
#     # # other
#     # "batch_norm": False,
#     # "dropout_rate": 0.1,
#     # "recon_loss_coef": 1,
#     # "kld_loss_coef": 0.1,
# }
# print("script")

# base_config = combine_with_defaults(base_config, defaults=vars(parse_args([])))


# params_grid = [{"target_hierarchy_level": [-1]}]

# print(base_config, params_grid)


# experiments_list = create_experiments_helper(
#     experiment_name=name,
#     project_name="multimodal/vaes",
#     script="python3 -u mrun.py",
#     python_path="",
#     exclude=[
#         "apptainer",
#         "venv",
#         "old_code",
#         "examples",
#     ],
#     tags=[name],
#     base_config=base_config,
#     params_grid=params_grid,
# )

# # REMEMBER THAT YOU NEED TO EXPORT NEPTUNE_API_TOKEN, NEPTUNE_PROJECT AND SET PYTHONPATH. Example:
# # export NEPTUNE_API_TOKEN="your_token"
# # export NEPTUNE_PROJECT="multimodal/vaes"
# # export PYTHONPATH=$PYTHONPATH:$(pwd)
