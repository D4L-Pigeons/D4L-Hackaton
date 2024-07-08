from typing import Generator
from utils.paths import CONFIG_PATH, RESULTS_PATH
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

# from utils.metrics import calculate_metrics, latent_metrics


def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Validate model")
    parser.add_argument(
        "--method",
        choices=["omivae", "babel", "advae", "vae"],
        help="Name of the method to use.",
    )
    parser.add_argument(
        "--cv-seed", default=42, help="Seed used to make k folds for cross validation."
    )
    parser.add_argument(
        "--n-folds", default=5, type=int, help="Number of folds in cross validation."
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

    return parser.parse_args()


def load_config(args) -> SimpleNamespace:
    def load_object(dct):
        return SimpleNamespace(**dct)

    with open(CONFIG_PATH / args.method / f"{args.config}.yaml") as file:
        config_dict = yaml.safe_load(file)
    config_namespace = json.loads(json.dumps(config_dict), object_hook=load_object)
    return config_namespace


def create_results_dir(args):
    """Creates directories for saving results."""
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    results_path = (
        RESULTS_PATH
        / args.method
        / f"{args.config}_{formatted_time}_{args.cv_seed}_seeds_{args.n_folds}_folds"
    )

    os.makedirs(results_path, exist_ok=True)  # Create directories if necessary

    return results_path


def save_model_and_latent(model, latent_representation, results_path):
    """Saves the configuration, model, and latent representation."""
    # Save config
    with open(results_path / "config.yaml", "w") as file:
        yaml.dump(model.config.__dict__, file)
        print(f"Config saved to: {results_path / 'config.yaml'}")

    # Save model
    saved_model_path = model.save(str(results_path / "saved_model"))
    print(f"Model saved to: {saved_model_path}")

    # Save latent
    if isinstance(latent_representation, dict):  # BABEL model
        for modality_name, latent in latent_representation.items():
            torch.save(latent, str(results_path / f"latent_train_{modality_name}"))
    else:
        torch.save(latent_representation, str(results_path / "latent"))


class CrossValidator:
    def __init__(self, model, random_state=41, n_folds=5, subsample_frac=None):
        self.model = model
        self.random_state = random_state
        self.n_folds = n_folds
        self.subsample_frac = subsample_frac

    def cross_validate_and_save(self, data, test_data, results_path):
        cv_metrics = self.cross_validate(data, test_data)
        print("Saving cross validation metrics to CSV...")
        cv_metrics.to_csv(str(results_path / "cv_metrics"))

    def cross_validate(self, data, test_data) -> pd.DataFrame:
        torch.manual_seed(self.random_state)
        cv_results = []
        for i, (train_data, val_data) in enumerate(
            _k_folds(data, self.n_folds, self.random_state, self.subsample_frac)
        ):
            self.model.fit(train_data)
            fold_metrics = self._evaluate(self.model, val_data)
            cv_results.append(fold_metrics)

        test_data_results = self._evaluate(self.model, test_data)

        return pd.DataFrame.from_dict(
            {"fold_wise": cv_results, "test_data": test_data_results}
        )

    def _evaluate(self, model, data):
        print(f"Evaluating model on fold...")
        return None  # Replace with actual metrics calculation

    def _k_folds(self, data: ad.AnnData) -> Generator[ad.AnnData, ad.AnnData]:
        r"""
        Generate indices for k-folds cross validation.

        Arguments:
            data (anndata.AnnData): The data to perform cross validation on.
            n_folds (int): Number of folds in cross validation.
            random_state (int): Seed used to make k folds for cross validation.

        Yields:
            Tuple[anndata.AnnData, anndata.AnnData]: The training and test data for each fold.
        """
        obs_names = data.obs_names.values
        if self.subsample_frac is not None:
            np.random.seed(self.random_state)
            obs_names = np.random.choice(
                obs_names, size=int(self.subsample_frac * len(obs_names)), replace=False
            )
        kfold = KFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )
        split = kfold.split(obs_names)

        for train_obs_names, val_obs_names in split:
            yield data[train_obs_names], data[val_obs_names]
