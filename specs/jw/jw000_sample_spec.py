from mrunner.helpers.specification_helper import create_experiments_helper

from src.global_utils.parser import combine_with_defaults

name = globals()["script"][:-3]


modalities_draft_config = {}

modalities_draft_config["gex"] = {
    # "modality_name": "GEX", # you can add modality name, but it's not necessary -- deafult name is name.upper()
    "dim": 13953,  # max 13953 | MAY BE CHANGED FOR DEBUGGING
}

modalities_draft_config["adt"] = {
    "dim": 134,  # max 134 | MAY BE CHANGED FOR DEBUGGING
}


base_config = {
    "method": "omivae",
    "model_name": "OmiAE",
    "mode": "train",
    "retrain": True,
    "lr": 0.001,
    "batch_size": 128,
    "subsample_frac": 1,  # MAY BE CHANGED FOR DEBUGGING
    "data_normalization": "standarize",  # "log1p", "standardize", "pearson_residuals", null -> None
    "remove_batch_effect": True,
    # include_class_labels: False
    "target_hierarchy_level": -1,
    "max_epochs": 10,
    "log_every_n_steps": 1,
    "early_stopping": True,
    "early_stopping_delta": 0.001,
    "early_stopping_patience": 5,
    "hidden_dim": 128,
    "encoder_hidden_dim": 128,
    "encoder_out_dim": 16,
    "latent_dim": 16,
    "decoder_hidden_dim": 128,
    # other
    "batch_norm": False,
    "dropout_rate": 0.1,
    "recon_loss_coef": 1,
    "kld_loss_coef": 0.1,
}

base_config = combine_with_defaults(base_config, modalities_draft_config)

params_grid = [{"target_hierarchy_level": [-1]}]

print(params_grid)


experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="multimodal/vaes",
    script="python3 -u mrun.py",
    python_path="",
    exclude=[
        "apptainer",
        "venv",
        "old_code",
        "examples",
    ],
    tags=[name],
    base_config=base_config,
    params_grid=params_grid,
)

# REMEMBER THAT YOU NEED TO EXPORT NEPTUNE_API_TOKEN, NEPTUNE_PROJECT AND SET PYTHONPATH. Example:
# export NEPTUNE_API_TOKEN="your_token"
# export NEPTUNE_PROJECT="multimodal/vaes"
# export PYTHONPATH=$PYTHONPATH:$(pwd)
