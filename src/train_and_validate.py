import argparse
import datetime
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import torch
import yaml
from sklearn.model_selection import KFold

from models.ModelBase import ModelBase
from utils.data_utils import load_anndata
from utils.paths import CONFIG_PATH, RESULTS_PATH


def main():
    parser = argparse.ArgumentParser(description="Validate model")
    parser.add_argument(
        "--method",
        choices=["omivae", "babel", "advae"],
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
        help="Name of a configuration in src/config/{method} directory.",
    )
    parser.add_argument(
        "--cv-seed", default=42, help="Seed used to make k folds for cross validation."
    )
    parser.add_argument(
        "--n-folds", default=5, help="Number of folds in cross validation."
    )
    parser.add_argument(
        "--retrain", default=True, help="Retrain a model using the whole dataset."
    )

    args = parser.parse_args()

    config = load_config(args)
    model = create_model(args, config)

    data = load_anndata(mode=args.mode)

    cross_validation_metrics = cross_validation(
        data, model, random_state=args.cv_seed, n_folds=args.n_folds
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

    # Save metrics
    cross_validation_metrics.to_json(results_path / "metrics.json", indent=4)
    print(f"Metrics saved to: {results_path / 'metrics.json'}")

    # Retrain and save model
    if args.retrain:
        print("Retraining model...")
        model.train(data)
        saved_model_path = model.save(str(results_path / "saved_model"))
        print(f"Model saved to: {saved_model_path}")


def load_config(args) -> argparse.Namespace:
    with open(CONFIG_PATH / args.method / f"{args.config}.yaml") as file:
        config = yaml.safe_load(file)

    return argparse.Namespace(**config)


def create_model(args, config) -> ModelBase:
    if args.method == "omivae":
        raise NotImplementedError(f"{args.method} method not implemented.")
    elif args.method == "babel":
        raise NotImplementedError(f"{args.method} method not implemented.")
    elif args.method == "advae":
        raise NotImplementedError(f"{args.method} method not implemented.")
    else:
        raise NotImplementedError(f"{args.method} method not implemented.")


def cross_validation(
    data: ad.AnnData, model: ModelBase, random_state: int = 42, n_folds: int = 5
) -> pd.DataFrame:
    torch.manual_seed(42)
    r"""
    Perform cross validation on the given data using the given model.

    Arguments:
        data (anndata.AnnData): The data to perform cross validation on.
        model (ModelBase): The model to validate.
        random_state (int): Seed used to make k folds for cross validation.
        n_folds (int): Number of folds in cross validation.

    Returns:
        pd.DataFrame: DataFrame containing the metrics for each fold and the average metrics.
    """

    metrics_names = [
        "f1_score_per_cell_type",
        "f1_score",
        "accuracy",
        "average_precision_per_cell_type",
        "roc_auc_per_cell_type",
        "confusion_matrix",
    ]

    cross_validation_metrics = pd.DataFrame(columns=metrics_names)

    for i, (train_data, test_data) in enumerate(k_folds(data, n_folds, random_state)):
        model.train(train_data)
        prediction = model.predict(test_data)
        prediction_probability = model.predict_proba(test_data)
        ground_truth = test_data.obs["cell_labels"]

        calculate_metrics(
            ground_truth,
            prediction,
            prediction_probability,
            test_data.obs["cell_labels"].cat.categories,
            cross_validation_metrics,
        )

        print(
            f"Validation accuracy of {i} fold:",
            cross_validation_metrics.loc[i]["accuracy"],
        )

    average_metrics = {
        metric_name: cross_validation_metrics[metric_name].mean()
        for metric_name in metrics_names
    }
    cross_validation_metrics.loc[len(cross_validation_metrics.index)] = average_metrics

    return cross_validation_metrics


def k_folds(data: ad.AnnData, n_folds: int, random_state: int):
    r"""
    Generate indices for k-folds cross validation.

    Arguments:
        data (anndata.AnnData): The data to perform cross validation on.
        n_folds (int): Number of folds in cross validation.
        random_state (int): Seed used to make k folds for cross validation.

    Yields:
        Tuple[anndata.AnnData, anndata.AnnData]: The training and test data for each fold.
    """
    sample_ids = data.obs["sample_id"].cat.remove_unused_categories()
    sample_ids_unique = sample_ids.cat.categories

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    split = kfold.split(sample_ids_unique.tolist())

    for train, test in split:
        train_mask = data.obs["sample_id"].isin(sample_ids_unique[train])
        test_mask = data.obs["sample_id"].isin(sample_ids_unique[test])

        yield data[train_mask], data[test_mask]


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


def calculate_metrics(
    ground_truth, prediction, prediction_probability, classes, cross_validation_metrics
):
    r"""
    Calculate metrics for a single fold.

    Arguments:
        ground_truth (array-like): Array of true labels.
        prediction (array-like): Array of predicted labels.
        prediction_probability (array-like): Array of predicted class probabilities.
        classes (array-like): Array of unique class labels.
        cross_validation_metrics (pd.DataFrame): DataFrame to store the metrics in.
    """
    f1_score_per_cell_type = sklearn.metrics.f1_score(
        ground_truth, prediction, labels=classes, average=None
    )
    f1_score = sklearn.metrics.f1_score(
        ground_truth, prediction, labels=classes, average="macro"
    )
    accuracy = sklearn.metrics.accuracy_score(ground_truth, prediction)
    if prediction_probability is not None:
        average_precision_per_cell_type = sklearn.metrics.average_precision_score(
            ground_truth, prediction_probability, average=None
        )
        roc_auc_per_cell_type = sklearn.metrics.roc_auc_score(
            ground_truth,
            prediction_probability,
            multi_class="ovr",
            average=None,
            labels=classes,
        )
    else:
        average_precision_per_cell_type = None
        roc_auc_per_cell_type = None
    confusion_matrix = sklearn.metrics.confusion_matrix(
        ground_truth, prediction, labels=classes
    )

    metrics = {
        "f1_score_per_cell_type": f1_score_per_cell_type,
        "f1_score": f1_score,
        "accuracy": accuracy,
        "average_precision_per_cell_type": average_precision_per_cell_type,
        "roc_auc_per_cell_type": roc_auc_per_cell_type,
        "confusion_matrix": confusion_matrix,
    }

    cross_validation_metrics.loc[len(cross_validation_metrics.index)] = metrics


if __name__ == "__main__":
    main()
