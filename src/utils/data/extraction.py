import h5py
from argparse import Namespace
from pandas import DataFrame
import numpy as np
import pandas as pd
from utils.data.hdf5 import decode_bytes


def get_dataset_obs(cfg: Namespace) -> DataFrame:
    r"""
    Extracts the dataset observations from an HDF5 file.

    Args:
        cfg (Namespace): The configuration object containing the path to the HDF5 file and the columns to extract.
            - path: A path to the hdf5 file.
            - obs.columns: A list of columns to be extracted from the obs object.

    Returns:
        DataFrame: The extracted dataset observations as a pandas DataFrame.
    """
    with h5py.File(cfg.path, "r") as f:
        obs = pd.DataFrame(
            data={col: np.array(f["obs"][col]) for col in cfg.obs.columns},
            index=decode_bytes(f["obs"]["_index"][:].astype(bytes)),
        )
    return obs
