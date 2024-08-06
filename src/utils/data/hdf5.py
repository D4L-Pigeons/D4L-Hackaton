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
