import argparse
import datetime
import json
import os
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import KFold

from models.babel import BabelModel
from models.ModelBase import ModelBase
from models.omivae_simple import OmiModel
from models.vae import VAE
from utils.data_utils import load_anndata
from utils.paths import CONFIG_PATH, RESULTS_PATH


def main():
    parser = argparse.ArgumentParser(description="Validate model")
    parser.add_argument(
        "--method",
        choices=["omivae", "babel", "advae", "vae"],
        help="Name of the method to use.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test", "train+test"],
        default="train",
        help="Mode to load the data for. Must be one of ['train', 'test', 'train+test'].",
    )
    parser.add_argument(
        "--plus-iid-holdout",
        action="store_true",
        help="Whether to include the iid_holdout data in the anndata object.",
    )
    parser.add_argument(
        "--config",
        default="standard",
        help="Name of a configuration in src/config/{method}.yaml.",
    )
    parser.add_argument(
        "--cv-seed", default=42, help="Seed used to make k folds for cross validation."
    )
    parser.add_argument(
        "--n-folds", default=5, type=int, help="Number of folds in cross validation."
    )
    parser.add_argument(
        "--preload-subsample-frac",
        default=None,
        type=float,
        help="Fraction of the data to load. If None, use all data. Don't use subsample-frac with this option.",
    )
    parser.add_argument(
        "--subsample-frac",
        default=None,
        type=float,
        help="Fraction of the data to use for cross validation. If None, use all data.",
    )
    parser.add_argument(
        "--retrain", default=True, help="Retrain a model using the whole dataset."
    )

    args = parser.parse_args()

    config = load_config(args)
    model = create_model(args, config)

    train_data = load_anndata(
        mode="train",
        normalize=config.normalize,
        remove_batch_effect=config.remove_batch_effect,
        target_hierarchy_level=config.target_hierarchy_level,
        preload_subsample_frac=args.preload_subsample_frac,
    )
    test_data = load_anndata(
        mode="test",
        normalize=config.normalize,
        remove_batch_effect=config.remove_batch_effect,
        target_hierarchy_level=config.target_hierarchy_level,
        preload_subsample_frac=args.preload_subsample_frac,
    )

    cross_validation_metrics = cross_validation(
        train_data,
        test_data,
        model,
        random_state=args.cv_seed,
        n_folds=args.n_folds,
        subsample_frac=args.subsample_frac,
    )

    # Save results
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Create directories if they don't exist
    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)

    results_path = RESULTS_PATH / args.method
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    results_path = (
        RESULTS_PATH
        / args.method
        / f"{args.config}_{formatted_time}_seed_{args.cv_seed}_folds_{args.n_folds}"
    )
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    # Save config
    with open(results_path / "config.yaml", "w") as file:
        yaml.dump(config.__dict__, file)
        print(f"Config saved to: {results_path / 'config.yaml'}")

    cross_validation_metrics.to_json(results_path / "metrics.json", indent=4)
    print(f"Metrics saved to: {results_path / 'metrics.json'}")

    # Retrain and save model
    if args.retrain:
        print("Retraining model...")
        full_data = load_anndata(
            mode="test",
            normalize=config.normalize,
            remove_batch_effect=config.remove_batch_effect,
            target_hierarchy_level=config.target_hierarchy_level,
            preload_subsample_frac=args.preload_subsample_frac,
        )
        model.fit(full_data)
        saved_model_path = model.save(str(results_path / "saved_model"))
        print(f"Model saved to: {saved_model_path}")


def load_config(args) -> SimpleNamespace:
    def load_object(dct):
        return SimpleNamespace(**dct)

    with open(CONFIG_PATH / args.method / f"{args.config}.yaml") as file:
        config_dict = yaml.safe_load(file)
    config_namespace = json.loads(json.dumps(config_dict), object_hook=load_object)
    return config_namespace
