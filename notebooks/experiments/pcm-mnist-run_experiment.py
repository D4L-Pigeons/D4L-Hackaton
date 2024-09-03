# %%
import os

if not "src" in os.listdir():
    os.chdir("../../")
os.listdir()

# %%
import torch
from pathlib import Path
import pytorch_lightning as pl

# Plotting
import matplotlib
import matplotlib.pyplot as plt

# import scienceplots

# plt.style.use("science")

# Neptune
from pytorch_lightning.loggers import NeptuneLogger

# Conditional MNIST dataset
from src.utils.config import load_config_from_path
from src.utils.data.pcm.mnist_cond_trans_dataset import (
    ConditionalMNIST,
    get_ConditionalMnistDataloader,
)

# Typing
from src.utils.common_types import Batch
from typing import Callable, Dict, Any, Tuple, List

# Other
import inspect
from functools import partial
from argparse import Namespace

# Paths
from src.utils.paths import CONFIG_PATH_DATA, CONFIG_PATH_MODELS

# Chain model
from src.models.components.chain import Chain

# Neptune
import neptune

# Plotting & Callbacks
from src.utils.evaluation.plots import (
    plot_original_vs_reconstructed,
    plot_images_with_conditions,
    wrap_with_first_batch,
    plot_latent,
    NeptunePlotLogCallback,
)

# %%

# %%
project_name = "multimodal/vaes"

# %%
# Access the NEPTUNE_API_TOKEN environment variable
neptune_api_token = os.getenv("NEPTUNE_API_TOKEN")

# Check if the token is available
if neptune_api_token is None:
    raise ValueError("NEPTUNE_API_TOKEN environment variable is not set.")

# %%
# from torch.utils.data import Subset

train_data_cfg_file_path = CONFIG_PATH_DATA / "pcm-mnist-02-train.yaml"

test_data_cfg_file_path = CONFIG_PATH_DATA / "pcm-mnist-02-test.yaml"

train_data_cfg = load_config_from_path(file_path=train_data_cfg_file_path)
cmnist_train = ConditionalMNIST(cfg=train_data_cfg)

# cmnist_train = Subset(cmnist_train, list(range(1000)))

cmnist_train_dataloader = get_ConditionalMnistDataloader(
    cmnist=cmnist_train, batch_size=128, shuffle=True
)

test_data_cfg = load_config_from_path(file_path=test_data_cfg_file_path)
cmnist_val = ConditionalMNIST(cfg=test_data_cfg)

# cmnist_val = Subset(cmnist_val, list(range(1000)))

cmnist_val_dataloader = get_ConditionalMnistDataloader(
    cmnist=cmnist_val, batch_size=128, shuffle=False
)

# %%
plot_images_with_conditions_wrapped_with_wrap_with_first_batch = wrap_with_first_batch(
    plot_images_with_conditions,
    **vars(
        Namespace(
            imgs_name="img",
            conditions_name="condition_token_ids",
            condition_values_name="condition_values",
            disp_batch_size=10,
            disp_n_latent_samples=1,
            filename_comp="filename",
            disp_img_size=2,
            y_title_shift=0.91,
        )
    ),
)

fig = plot_images_with_conditions_wrapped_with_wrap_with_first_batch(
    dataloader=cmnist_val_dataloader, processing_function=lambda batch: batch
)

fig

# %% [markdown]
# ### Plotting Callbacks

# %%
plot_prior_sampled_imgs_with_conditions_wrapped = wrap_with_first_batch(
    plot_images_with_conditions,
    **vars(
        Namespace(
            filename_comp="embeddings_plot",
            imgs_name="img",
            conditions_name="condition_token_ids",
            condition_values_name="condition_values",
            disp_batch_size=10,
            disp_n_latent_samples=16,
            disp_img_size=2,
            y_title_shift=0.91,
        )
    ),
)

plot_sample_prior_callback = NeptunePlotLogCallback(
    plotting_function_taking_dataloader=plot_prior_sampled_imgs_with_conditions_wrapped,
    command_name="sample-prior",
    neptune_plot_log_path="validation_plots/sample_prior",
    plot_file_base_name="sample_prior",
    command_dynamic_kwargs={},
)

# %%
plot_original_vs_reconstructed_wrapped = wrap_with_first_batch(
    plot_original_vs_reconstructed,
    **vars(
        Namespace(
            org_imgs_name="img_org",
            reconstructed_imgs_name="img",
            num_images=10,
            wspace=0.25,
            hspace=0.25,
            filename_comp="org_vs_reconstr",
        )
    ),
)

plot_reconstruction_callback = NeptunePlotLogCallback(
    plotting_function_taking_dataloader=plot_original_vs_reconstructed_wrapped,
    command_name="encode-decode",
    neptune_plot_log_path="validation_plots/reconstructed",
    plot_file_base_name="embedding",
    command_dynamic_kwargs={},
)

# %%
plot_embeddings_callback = NeptunePlotLogCallback(
    plotting_function_taking_dataloader=partial(
        plot_latent,
        **vars(
            Namespace(
                data_name="img",
                condition_value_name="condition_values",
                filename_comp="latent",
                condition_value_idxs=[0, 1, 2, 3, 4],
                are_conditions_categorical=[True, True, True, True, True],
            )
        ),
    ),
    command_dynamic_kwargs={},
    command_name="encode",
    neptune_plot_log_path="validation_plots/embeddings",
    plot_file_base_name="embedding",
)

# %% [markdown]
# ## Model

# %%
model_cfg_file_path = CONFIG_PATH_MODELS / "pcm-04.yaml"

chain_cfg = load_config_from_path(file_path=model_cfg_file_path)
chainae = Chain(cfg=chain_cfg)
chainae

# %%
# Create a Neptune logger
neptune_logger = NeptuneLogger(
    api_key=neptune_api_token,
    project=project_name,
    name="cpiwae-3-09-24",
)

trainer = pl.Trainer(
    max_epochs=500,
    logger=neptune_logger,
    check_val_every_n_epoch=10,
    callbacks=[
        plot_sample_prior_callback,
        plot_embeddings_callback,
        plot_reconstruction_callback,
    ],
)

# %%
if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
    trainer.logger.experiment[f"config/model_config.yaml"].upload(
        neptune.types.File(str(model_cfg_file_path))
    )
    trainer.logger.experiment[f"config/train_data_config.yaml"].upload(
        neptune.types.File(str(train_data_cfg_file_path))
    )
    trainer.logger.experiment[f"config/test_data_config.yaml"].upload(
        neptune.types.File(str(test_data_cfg_file_path))
    )

# %%
trainer.fit(
    model=chainae,
    train_dataloaders=cmnist_train_dataloader,
    val_dataloaders=cmnist_val_dataloader,
)

# %%
# Stop the Neptune experiment after training ends
neptune_logger.experiment.stop()

# %%
# run_id = "VAES-40"  # Replace with your run ID
# run = neptune.init_run(
#     project=project_name, api_token=neptune_api_token, with_id=run_id, mode="read-only"
# )

# %%
# run.get_structure()

# %%
# run["training/model/checkpoints/epoch=9-step=240"].download()
