# calculates desired metrics
# --- METRICS ---


def _calc_entropy(preds_probs: ndarray) -> float:
    r"""
    Calculates the entropy of a distribution.

    Parameters:
    preds_probs (ndarray): The predicted probabilities of the distribution.

    Returns:
    float: The entropy value.

    """
    return -np.sum(preds_probs * np.log(preds_probs + 1e-10), axis=1).mean()


def calc_metrics_from_uniport(
    train_embeddings: ndarray,
    generalization_embeddings: ndarray,
    train_df: DataFrame,
    generalization_df: DataFrame,
    n_neighbors: int = 50,
    n_loc_samples: int = 100,
) -> DataFrame:
    r"""
    Calculates the metrics from the embeddings and the metadata.

    Args:
        train_embeddings (ndarray): Array of training embeddings.
        generalization_embeddings (ndarray): Array of generalization embeddings.
        train_df (DataFrame): DataFrame containing training metadata.
        generalization_df (DataFrame): DataFrame containing generalization metadata.
        n_neighbors (int, optional): Number of neighbors to consider for KNN classification. Defaults to 50.
        n_loc_samples (int, optional): Number of samples to select from generalization embeddings. Defaults to 100.

    Returns:
        DataFrame: DataFrame containing calculated metrics for each column in train_df.

    Note:
        AverageFOSCTTM is not calculated.
    """
    knc = KNeighborsClassifier(n_neighbors=n_neighbors)

    metrics = pd.DataFrame(
        columns=[
            "col_name",
            "AdjustedRandIndex",
            "NormalizedMutualInformation",
            "F1Score",
            "SilhouetteCoefficient",
            "MixingEntropyScore",
        ],
        index=train_df.columns,
    )
    sampled_region_indices = torch.randint(
        0, generalization_embeddings.shape[0], (n_loc_samples,)
    )
    for col_name in train_df.columns:
        knc.fit(train_embeddings, train_df[col_name])
        labels = generalization_df[col_name]
        preds = knc.predict(generalization_embeddings)
        preds_probs = knc.predict_proba(
            generalization_embeddings[sampled_region_indices, :]
        )
        metrics.loc[col_name] = [
            col_name,
            adjusted_rand_score(labels, preds),
            normalized_mutual_info_score(labels, preds),
            f1_score(labels, preds, average="weighted"),
            (silhouette_score(generalization_embeddings, labels) + 1)
            * 0.5,  # Range [0-1] instead of [-1, 1]
            _calc_entropy(preds_probs),
        ]

    return metrics
