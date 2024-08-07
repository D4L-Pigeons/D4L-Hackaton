from sklearn.model_selection import KFold, train_test_split
from pandas import DataFrame
import numpy as np
from numpy import ndarray
from argparse import Namespace
from typing import List, Tuple, Dict, Generator, TypeAlias, NamedTuple


class Split(NamedTuple):
    train_indices: ndarray
    val_indices: ndarray


def naive_mixing_fraction_split(max_idx: int, cfg: Namespace) -> Split:
    r"""
    Splits the data into training and validation sets using a naive mixing fraction approach.

    Parameters:
        max_idx (int): The maximum index of the data.
        cfg (Namespace): The configuration object containing the validation fraction.
            - val_fraction: The fraction of validation data in the split.

    Returns:
        Generator[Tuple[ndarray, ndarray]]: A generator that yields tuples of training and validation sets.
    """
    return Split(*train_test_split(np.arange(max_idx), test_size=cfg.val_fraction))


def naive_mixing_k_fold_split(
    max_idx: int, cfg: Namespace
) -> Generator[Split, None, None]:
    r"""
    Perform a naive mixing k-fold split on the given data.

    Parameters:
    max_idx (int): The maximum index of the data.
    cfg (Namespace): The configuration object containing the number of splits.
        - n_splits: The number of splits in CV.
        - random_state: The random_state argument ot KFold.

    Returns:
    - The indices for the first and second sets for each fold.
    """

    kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
    for split in kf.split(np.arange(max_idx)):
        yield Split(*split)


# Named for lack of better words. It does not ensure any specified proportion of the cells in different datasets.
def composite_k_fold_split(
    df: DataFrame, cfg: Namespace
) -> Generator[Split, None, None]:
    r"""
    Splits the DataFrame into train and validation sets using composite k-fold cross-validation.

    Args:
        df (DataFrame): The input DataFrame to be split.
        cfg (Namespace): The configuration object containing the parameters for the split.
            - grid_variables: The list of variables defining a crossvalidation grid.
            - n_splits: The number of splits in CV.
            - random_state: The random_state argument ot KFold.

    Returns:
        Generator: A generator that yields tuples of train and validation sets.

    """
    df["index"] = np.arange(len(df))
    df = df.set_index(keys=cfg.grid_variables, inplace=False)
    present_value_combinations = df.index.values
    splits = KFold(
        n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state
    ).split(np.arange(len(present_value_combinations)))
    for split in splits:
        train_indices, val_indices = split
        yield Split(
            train_indices=df.iloc[train_indices]["index"].values,
            val_indices=df.iloc[val_indices]["index"].values,
        )


def subset_parameterised_composite_split(
    df: DataFrame,
    cfg: Namespace,
) -> Tuple[List[int], List[int]]:
    r"""
    Splits the indices into two parts based on a composite condition.

    Args:
        df (DataFrame): The input DataFrame to be split.
        cfg (Namespace): The configuration object containing the necessary parameters.
            - val_filter_values List[int]: A list of variables used to create a grid and corresponding filtering values.

    Returns:
        Tuple[List[int], List[int]]: A tuple containing two lists of indices. The first list
        contains the indices where the composite condition is False, and the second list
        contains the indices where the composite condition is True.
    """
    split_mask = (
        df[
            [filter_var_dict["name"] for filter_var_dict in cfg.val_filter_values]
        ]  # The maching order of columns between config and the df is ensured here.
        .apply(
            lambda x: all(
                value in filter_var_dict["filter_values"]
                for value, filter_var_dict in zip(x.values, cfg.val_filter_values)
            ),
            axis=1,
        )
        .values
    )
    indices = np.arange(len(df))
    assert (
        split_mask.sum() > 0
    ), "There are selected indices for the validation set. Check the df cfg.val_filtere_values for a type mismatch or lack of appropriate combinations."
    return Split(
        train_indices=indices[np.logical_not(split_mask)],
        val_indices=indices[split_mask],
    )
