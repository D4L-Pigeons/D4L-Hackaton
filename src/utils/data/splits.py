from sklearn.model_selection import KFold, train_test_split
from pandas import DataFrame
import numpy as np
from numpy import ndarray
from argparse import Namespace
from typing import List, Tuple, Dict, Generator


def subset_parameterised_composite_split(
    df: DataFrame,
    val_filter_values: List[int],
    variables_metadata: Dict[str, Namespace],
    cfg: Namespace,
) -> Tuple[List[int], List[int]]:
    r"""
    Splits the indices into two parts based on a composite condition.

    Args:
        df (DataFrame): The input DataFrame to be split.
        cfg (Namespace): The configuration object containing the necessary parameters.
            - grid_variables: A list of variables used to create a grid.

    Returns:
        Tuple[List[int], List[int]]: A tuple containing two lists of indices. The first list
        contains the indices where the composite condition is False, and the second list
        contains the indices where the composite condition is True.
    """
    split_mask = (
        df[cfg.grid_variables]
        .apply(
            lambda x: all(
                value in var_meta.categories[vfv]
                for value, vfv, var_meta in zip(
                    x.values, val_filter_values, variables_metadata
                )
            ),
            axis=1,
        )
        .values
    )
    indices = np.arange(len(df))
    return indices[np.logical_not(split_mask)], indices[split_mask]


def naive_mixing_fraction_split(
    max_idx: int, cfg: Namespace
) -> Generator[Tuple[ndarray, ndarray], None, None]:
    r"""
    Splits the data into training and validation sets using a naive mixing fraction approach.

    Parameters:
        max_idx (int): The maximum index of the data.
        cfg (Namespace): The configuration object containing the validation fraction.
            - val_fraction: The fraction of validation data in the split.

    Returns:
        Generator[Tuple[ndarray, ndarray]]: A generator that yields tuples of training and validation sets.
    """
    return train_test_split(np.arange(max_idx), test_size=cfg.val_fraction)


def naive_mixing_k_fold_split(
    max_idx: int, cfg: Namespace
) -> Generator[Tuple[ndarray, ndarray], None, None]:
    r"""
    Perform a naive mixing k-fold split on the given data.

    Parameters:
    max_idx (int): The maximum index of the data.
    cfg (Namespace): The configuration object containing the number of splits.
        - n_splits: The number of splits in CV.

    Returns:
    - The indices for the first and second sets for each fold.
    """

    kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
    return kf.split(np.arange(max_idx))


# Named for lack of better words. It does not ensure any specified proportion of the cells in different datasets.
def composite_k_fold_split(df: DataFrame, cfg: Namespace) -> Generator:
    r"""
    Splits the DataFrame into train and validation sets using composite k-fold cross-validation.

    Args:
        df (DataFrame): The input DataFrame to be split.
        cfg (Namespace): The configuration object containing the parameters for the split.
            - grid_variables: The list of variables defining a crossvalidation grid.
            - n_splits: The number of splits in CV.

    Returns:
        Generator: A generator that yields tuples of train and validation sets.

    """
    df["index"] = np.arange(len(df))
    df.set_index(keys=cfg.grid_variables, inplace=True)
    present_value_combinations = df.index.values
    splits = KFold(
        n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state
    ).split(present_value_combinations)
    return (
        (df.loc[train_indices]["index"].values, df.loc[val_indices]["index"].values)
        for train_indices, val_indices in splits
    )
