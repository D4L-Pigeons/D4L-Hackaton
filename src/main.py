import os
from pathlib import Path
import torch
from pathlib import Path
import pytorch_lightning as pl

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
    plot_latent_with_pca_umap,
)

# Checking file run location.
if Path(os.getcwd()).name != "D4L-Hackaton":
    raise ValueError(
        "Incorrect directory. Please make sure you are in the 'D4L-Hackaton' directory."
    )

# Access the NEPTUNE_API_TOKEN environment variable
neptune_api_token = os.getenv("NEPTUNE_API_TOKEN")
if neptune_api_token is None:
    raise ValueError("NEPTUNE_API_TOKEN environment variable is not set.")
