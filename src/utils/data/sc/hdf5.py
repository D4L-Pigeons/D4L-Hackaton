import h5py
import numpy as np
from numpy import ndarray


def decode_bytes(bytes: ndarray[np.byte]) -> ndarray[str]:
    f"""
    Decodes a numpy array of bytes into a numpy array of strings using UTF-8 encoding.

    Parameters:
    bytes (ndarray[np.byte]): The numpy array of bytes to be decoded.

    Returns:
    ndarray[str]: The numpy array of decoded strings.
    """
    return np.char.decode(bytes, "utf-8")


def print_hdf5_structure_tree(name: str, obj):
    r"""
    Recursively prints the tree structure of an HDF5 file or group.

    Parameters:
    - name (str): The name of the current object.
    - obj (h5py.Group or h5py.Dataset): The object to print.

    Returns:
    None
    """
    print(name)
    if isinstance(obj, h5py.Group):
        print(f"{name}")
        for key, val in obj.items():
            print_hdf5_structure_tree(f"{name}\{key}", val)


# def print_tree(name, obj):
#     print(name)
#     if isinstance(obj, h5py.Group):
#         for key, val in obj.items():
#             print_tree(f"{name}/{key}", val)


# def print_datasets(name, obj):

#     if isinstance(obj, h5py.Dataset):
#         print(name)
