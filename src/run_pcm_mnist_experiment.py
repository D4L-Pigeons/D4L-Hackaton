import os
import signal
import sys
import argparse
from pathlib import Path

# Neptune
import neptune
from pytorch_lightning.loggers import NeptuneLogger

# Conditional MNIST dataset
from utils.config import load_config_from_path
from utils.data.pcm.mnist_cond_trans_dataset import (
    ConditionalMNIST,
    get_ConditionalMnistDataloader,
)

# Other
from functools import partial
from argparse import Namespace

# Paths
from utils.paths import CONFIG_PATH_DATA, CONFIG_PATH_MODELS, CONFIG_PATH_TRAINER

# Chain model
from models.components.chain import Chain

# Trainer
from utils.pl_trainer.trainer import get_trainer


DEFAULT_PROJECT_NAME: str = "multimodal/vaes"
DEFAULT_BATCH_SIZE: int = 128


def get_parser():
    parser = argparse.ArgumentParser(description="Run PCM MNIST Experiment")

    data = parser.add_argument_group("Data configuration")
    data.add_argument(
        "--data_train",
        type=str,
        required=True,
        help="Filename of the training data configuration file",
    )
    data.add_argument(
        "--data_val",
        type=str,
        required=True,
        help="Filename of the validation data configuration file.",
    )
    data.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training and validation",
    )
    data.add_argument(
        "--subsample_size",
        type=int,
        help="Subsample size for the dataset",
        default=None,
    )

    experiment = parser.add_argument_group("Experiment configuration")
    experiment.add_argument(
        "--project_name",
        type=str,
        default=DEFAULT_PROJECT_NAME,
        help="Neptune project name",
    )
    experiment.add_argument(
        "--experiment_name", type=str, required=True, help="Name of the experiment"
    )

    parser.add_argument(
        "--model_cfg", type=str, required=True, help="Filename of the model config file"
    )

    trainer = parser.add_argument_group("Trainer configuration")
    trainer.add_argument(
        "--trainer_cfg",
        type=str,
        help="Filename of the trainer configuration file.",
    )
    trainer.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Maximum number of training epochs",
    )
    trainer.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=None,
        help="Number of epochs between each validation check",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Weather to present the status of script run.",
    )

    return parser


def signal_handler(sig, frame):
    print("Interrupt received, stopping experiment...")
    if "neptune_logger" in globals():
        neptune_logger.experiment.stop()
    sys.exit(0)


def main(args: Namespace) -> None:

    if args.verbose:
        print("Checking file run location.")

    if Path(os.getcwd()).name != "D4L-Hackaton":
        raise ValueError(
            "Incorrect directory. Please make sure you are in the 'D4L-Hackaton' directory."
        )

    if args.verbose:
        print("Accessing the NEPTUNE_API_TOKEN environment variable.")

    NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN")
    if NEPTUNE_API_TOKEN is None:
        raise ValueError("NEPTUNE_API_TOKEN environment variable is not set.")

    if args.verbose:
        print("Setting up datasets.")

    train_data_cfg_file_path: Path = CONFIG_PATH_DATA / args.data_train
    train_data_cfg = load_config_from_path(file_path=train_data_cfg_file_path)
    cmnist_train = ConditionalMNIST(cfg=train_data_cfg)

    val_data_cfg_file_path: Path = CONFIG_PATH_DATA / args.data_val
    val_data_cfg = load_config_from_path(file_path=val_data_cfg_file_path)
    cmnist_val = ConditionalMNIST(cfg=val_data_cfg)

    # For debugging a subsample_size may be specified for faster processing.
    if args.subsample_size is not None:
        if args.verbose:
            print("Subsampling.")

        from torch.utils.data import Subset

        lower_bound = min(len(cmnist_train), len(cmnist_val))
        assert (
            args.subsample_size <= lower_bound
        ), f"The subsample_size must be lower or equal to {lower_bound}."

        subset_idxs = list(range(args.subsample_size))
        cmnist_train = Subset(cmnist_train, subset_idxs)
        cmnist_val = Subset(cmnist_val, subset_idxs)

    if args.verbose:
        print("Setting up dataloaders.")

    cmnist_train_dataloader = get_ConditionalMnistDataloader(
        cmnist=cmnist_train, batch_size=args.batch_size, shuffle=True
    )
    cmnist_val_dataloader = get_ConditionalMnistDataloader(
        cmnist=cmnist_val, batch_size=args.batch_size, shuffle=False
    )

    if args.verbose:
        print("Setting up model.")

    model_cfg_file_path: Path = CONFIG_PATH_MODELS / args.model_cfg
    chain_cfg = load_config_from_path(file_path=model_cfg_file_path)
    chain = Chain(cfg=chain_cfg)

    if args.verbose:
        print("Setting up neptune logger.")

    global neptune_logger
    neptune_logger = NeptuneLogger(
        api_key=NEPTUNE_API_TOKEN,
        project=args.project_name,
        name=args.experiment_name,
    )

    if neptune_logger is not None and hasattr(neptune_logger, "experiment"):
        neptune_logger.experiment[f"config/model_config.yaml"].upload(
            neptune.types.File(str(model_cfg_file_path))
        )
        neptune_logger.experiment[f"config/train_data_config.yaml"].upload(
            neptune.types.File(str(train_data_cfg_file_path))
        )
        neptune_logger.experiment[f"config/val_data_config.yaml"].upload(
            neptune.types.File(str(val_data_cfg_file_path))
        )
        neptune_logger.experiment[f"config/val_data_config.yaml"].upload(
            neptune.types.File(str(val_data_cfg_file_path))
        )

    trainer_cfg_path: Path = CONFIG_PATH_TRAINER / args.trainer_cfg
    trainer_cfg = load_config_from_path(file_path=trainer_cfg_path)

    if args.verbose:
        print("Setting up trainer.")

    trainer = get_trainer(
        cfg=trainer_cfg,
        logger=neptune_logger,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )

    if args.verbose:
        print("Training...")

    trainer.fit(
        model=chain,
        train_dataloaders=cmnist_train_dataloader,
        val_dataloaders=cmnist_val_dataloader,
    )

    if args.verbose:
        print("Training stopped.")

    neptune_logger.experiment.stop()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    parser = get_parser()
    try:
        args, unknown_args = parser.parse_known_args()
        if unknown_args:
            print(f"Non-parsed arguments: {unknown_args}")
    except SystemExit as e:
        print(f"Error parsing arguments: {e}")
        parser.print_help()
        sys.exit(2)

    main(args)
