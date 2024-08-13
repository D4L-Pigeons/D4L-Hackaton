import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    f1_score,
    silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier
import torch


def _calc_entropy(preds_probs: np.ndarray) -> float:
    r"""
    Calculates the entropy of a distribution.

    Parameters:
    preds_probs (ndarray): The predicted probabilities of the distribution.

    Returns:
    float: The entropy value.

    """
    return -np.sum(preds_probs * np.log(preds_probs + 1e-10), axis=1).mean()


def calc_metrics_from_uniport(
    train_embeddings: np.ndarray,
    generalization_embeddings: np.ndarray,
    train_df: pd.DataFrame,
    generalization_df: pd.DataFrame,
    n_neighbors: int = 50,
    n_loc_samples: int = 100,
    metrics_list: List[str] = [
        "AdjustedRandIndex",
        "NormalizedMutualInformation",
        "F1Score",
        "SilhouetteCoefficient",
        "MixingEntropyScore",
    ],
) -> pd.DataFrame:
    r"""
    Calculates the metrics from the embeddings and the metadata.

    Args:
        train_embeddings (ndarray): Array of training embeddings.
        generalization_embeddings (ndarray): Array of generalization embeddings.
        train_df (DataFrame): DataFrame containing training metadata.
        generalization_df (DataFrame): DataFrame containing generalization metadata.
        n_neighbors (int, optional): Number of neighbors to consider for KNN classification. Defaults to 50.
        n_loc_samples (int, optional): Number of samples to select from generalization embeddings. Defaults to 100.
        metrics_list (List[str], optional): List of metrics to calculate. Defaults to all metrics.

    Returns:
        DataFrame: DataFrame containing calculated metrics for each column in train_df.

    Note:
        AverageFOSCTTM is not calculated.
    """

    # Ensure the number of components isn't larger than the data
    assert (
        n_loc_samples <= generalization_embeddings.shape[0]
    ), "n_loc_samples cannot exceed the number of generalization samples."

    knc = KNeighborsClassifier(n_neighbors=n_neighbors)
    metrics = pd.DataFrame(columns=metrics_list, index=train_df.columns)

    sampled_region_indices = np.random.randint(
        0, generalization_embeddings.shape[0], n_loc_samples
    )

    for col_name in train_df.columns:
        knc.fit(train_embeddings, train_df[col_name])
        labels = generalization_df[col_name]
        preds = knc.predict(generalization_embeddings)
        preds_probs = knc.predict_proba(
            generalization_embeddings[sampled_region_indices, :]
        )

        if "AdjustedRandIndex" in metrics_list:
            metrics.at[col_name, "AdjustedRandIndex"] = adjusted_rand_score(
                labels, preds
            )
        if "NormalizedMutualInformation" in metrics_list:
            metrics.at[col_name, "NormalizedMutualInformation"] = (
                normalized_mutual_info_score(labels, preds)
            )
        if "F1Score" in metrics_list:
            metrics.at[col_name, "F1Score"] = f1_score(
                labels, preds, average="weighted"
            )
        if "SilhouetteCoefficient" in metrics_list:
            metrics.at[col_name, "SilhouetteCoefficient"] = (
                silhouette_score(generalization_embeddings, labels) + 1
            ) * 0.5
        if "MixingEntropyScore" in metrics_list:
            metrics.at[col_name, "MixingEntropyScore"] = _calc_entropy(preds_probs)

    return metrics
