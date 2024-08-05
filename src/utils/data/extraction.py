import h5py
from argparse import Namespace
from pandas import DataFrame
import numpy as np
import pandas as pd


def get_dataset_obs(cfg: Namespace) -> DataFrame:
    r"""
    Extracts the dataset observations from an HDF5 file.

    Args:
        cfg (Namespace): The configuration object containing the path to the HDF5 file and the columns to extract.

    Returns:
        DataFrame: The extracted dataset observations as a pandas DataFrame.
    """
    with h5py.File(cfg.path, "r") as f:
        obs = pd.DataFrame(
            data={col: np.array(f["obs"][col]) for col in cfg.columns},
            index=np.char.decode(
                np.array(f["obs"]["_index"], dtype=np.bytes_), "utf-8"
            ),
        )
    return obs
