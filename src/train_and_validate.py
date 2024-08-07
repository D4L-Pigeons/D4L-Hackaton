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
from utils.metrics import calculate_metrics, latent_metrics
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


def create_model(args, config) -> ModelBase:
    if args.method == "omivae":
        model = OmiModel(config)
        return model
    elif args.method == "babel":
        model = BabelModel(config)
        return model
    elif args.method == "advae":
        raise NotImplementedError(f"{args.method} method not implemented.")
    elif args.method == "vae":
        return VAE(config)
    else:
        raise NotImplementedError(f"{args.method} method not implemented.")


def cross_validation(
    data: ad.AnnData,
    test_data: ad.AnnData,
    model: ModelBase,
    random_state: int = 42,
    n_folds: int = 5,
    subsample_frac: float | None = None,
) -> pd.DataFrame:
    torch.manual_seed(42)
    r"""
    Perform cross validation on the given data using the given model.

    Arguments:
        data (anndata.AnnData): The data to perform cross validation on.
        test_data (anndata.AnnData): The data to.
        model (ModelBase): The model to validate.
        random_state (int): Seed used to make k folds for cross validation.
        n_folds (int): Number of folds in cross validation.

    Returns:
        pd.DataFrame: DataFrame containing the metrics for each fold and the average metrics.
    """

    for i, (train_data, val_data) in enumerate(
        k_folds(data, n_folds, random_state, subsample_frac)
    ):
        model.fit(train_data)
        latent_metrics(model, val_data)

    print("Calculate metrics...")

    metrics_dict = calculate_metrics(model, data, test_data)

    return pd.DataFrame.from_dict(metrics_dict)


def k_folds(
    data: ad.AnnData,
    n_folds: int,
    random_state: int,
    subsample_frac: float | None = None,
):
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
    if subsample_frac is not None:
        np.random.seed(random_state)
        obs_names = np.random.choice(
            obs_names, size=int(subsample_frac * len(obs_names)), replace=False
        )
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    split = kfold.split(obs_names)

    for train_obs_names, val_obs_names in split:
        yield data[train_obs_names], data[val_obs_names]


# def macro_average_precision(ground_truth, prediction_probability):
#   """
#   Calculates macro-averaged precision for multi-class classification.

#   Args:
#       ground_truth (array-like): Array of true labels.
#       prediction_probability (array-like): Array of predicted class probabilities.

#   Returns:
#       float: Macro-averaged precision score.
#   """
#   num_classes = len(ground_truth.unique())

#   precision_per_class = []

#   # Calculate precision score for each class
#   for class_label in range(num_classes):
#     class_mask = ground_truth == class_label
#     ground_truth_filtered = ground_truth[class_mask]
#     prediction_probability_filtered = prediction_probability[class_mask]
#     # Calculate precision for this class
#     precision = sklearn.metrics.precision_score(ground_truth_filtered, prediction_probability_filtered[:, class_label], average='binary', zero_division=0)
#     precision_per_class.append(precision)

#   # Macro-average the precision scores
#   macro_average_precision = np.mean(precision_per_class)
#   return macro_average_precision

if __name__ == "__main__":
    main()
