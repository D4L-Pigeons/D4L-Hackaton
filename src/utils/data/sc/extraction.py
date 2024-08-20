from argparse import Namespace
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from src.utils.data.sc.hdf5 import decode_bytes


def get_dataset_obs(cfg: Namespace, indices: Optional[ndarray] = None) -> DataFrame:
    r"""
    Extracts the dataset observations from an HDF5 file.

    Args:
        cfg (Namespace): The configuration object containing the path to the HDF5 file and the columns to extract.
            - path: A path to the hdf5 file.
            - obs.columns: A list of columns to be extracted from the obs object.
            - obs.columns.as_codes:

    Returns:
        DataFrame: The extracted dataset observations as a pandas DataFrame.
    """
    with h5py.File(cfg.path, "r") as f:
        index = (
            decode_bytes(f["obs"]["_index"][indices].astype(bytes))
            if indices is not None
            else decode_bytes(f["obs"]["_index"][:].astype(bytes))
        )
        data = {}
        for col in cfg.obs.columns:
            codes = (
                np.array(f["obs"][col["name"]][indices])
                if indices is not None
                else np.array(f["obs"][col["name"]][:])
            )
            if col["as_codes"]:
                data[col["name"]] = codes
            else:  # Original category names.
                data[col["name"]] = decode_bytes(
                    f["obs"]["__categories"][col["name"]][:][codes].astype(bytes)
                )

            # data = {
            #     # col["name"]: (
            #         # np.array(f["obs"][col["name"]][indices])
            #         # if indices is not None
            #         else (
            #             np.array(f["obs"][col["name"]][:])
            #             if col["as_codes"]
            #             else decode_bytes(
            #                 f["obs"]["__categories"][col["name"]][indices].astype(bytes)
            #                 if indices is not None
            #                 else f["obs"]["__categories"][col["name"]][:].astype(bytes)
            #             )[
            #                 (
            #                     np.array(f["obs"][col["name"]][indices])
            #                     if indices is not None
            #                     else np.array(f["obs"][col["name"]][:])
            #                 )
            #             ]
            #         )
            #     )
            #     for col in cfg.obs.columns
            # }
        obs = pd.DataFrame(
            data=data,
            index=index,
        )
    return obs
