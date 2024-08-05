import torch
from torch import Tensor
from typing import Dict
from argparse import Namespace
import h5py
from scipy.sparse import csr_matrix
from utils.paths import RAW_DATA_PATH, PREPROCESSED_DATA_PATH

# NOTES
# As for now the preprocessing operated on the whole dataset. "Running" version of the preprocessing migh be required for the dataset which do not fit into memory. Then statistics used for preprocessing need to be calculated in a running manner.


def _divide_by_nonzero_median(x: Tensor) -> Tensor:
    r"""
    Divides each element of the input tensor `x` by the median of non-zero elements along each column.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Tensor with each element divided by the median of non-zero elements along each column.
    """
    nonzero_mask = x > 0
    n_row_nonzero = nonzero_mask.sum(
        dim=0, keepdim=True
    )  # Calculate the count of nonzero entires in each column.
    nonzero_medians = x.sort(dim=0, descending=True)[0][
        :, torch.cat([(n_row_nonzero - 1) // 2, n_row_nonzero // 2], dim=1)
    ].mean(
        dim=1, keepdim=True
    )  # Calculate the median for each column.
    output = x / nonzero_medians
    return output


_TRANSFORMS_DICT: Dict = {
    "normalize_total": lambda x: x / x.sum(dim=1, keepdim=True),
    "divide_by_nonzero_median": _divide_by_nonzero_median,
    "log1p": lambda x: torch.log1p(x),
}


def _transform_tensor(transform_key: str, x: Tensor) -> Tensor:
    r"""
    Apply a transformation to the input tensor.

    Args:
        transform_key (str): The key representing the transformation to be applied.
        x (Tensor): The input tensor to be transformed.

    Returns:
        Tensor: The transformed tensor.

    Raises:
        KeyError: If the transform_key is not found in the __TRANSFORMS_DICT.

    """
    return _TRANSFORMS_DICT[transform_key](x)


def preprocess_and_save_dataset(
    cfg: Namespace,
) -> None:
    r"""
    Preprocesses the data and saves it to an HDF5 file.

    Args:
        cfg (Namespace): The configuration object containing preprocessing options.
            - read_file (Namespace): The configuration object for the input file.
                - filename (str): The name of the input file.
            - write_file (Namespace): The configuration object for the output file.
                - filename (str): The name of the output file.
    Returns:
        None
    """
    read_path = RAW_DATA_PATH / cfg.read_file.filename
    assert (
        read_path.exists()
    ), f"The file poited to by a path='{read_path}' does not exist."
    write_path = PREPROCESSED_DATA_PATH / cfg.write_file.filename
    if not write_path.exists():
        write_path.touch()

    with h5py.File(read_path, mode="r") as read_file, h5py.File(
        write_path, "w"
    ) as write_file:

        for key in read_file.keys():
            if key in cfg.groups_to_clone:
                read_file.copy(key, write_file)

        data = read_file["X/data"][:]
        indices = read_file["X/indices"][:]
        indptr = read_file["X/indptr"][:]
        shape = read_file["X"].attrs[
            "shape"
        ]  # Assuming the shape is stored as an attribute

        sparse_matrix = csr_matrix((data, indices, indptr), shape=shape)
        # Convert the sparse matrix to a PyTorch tensor
        tensor_data = torch.tensor(sparse_matrix.toarray())
        # tensor_data = torch.tensor(read_file["X"].toarray())

        # Apply transformations in specified order
        for transform_key in cfg.transforms:
            tensor_data = _transform_tensor(transform_key, tensor_data)

        # Go back to sparse format
        sparse_tensor = tensor_data.to_sparse()
        indices = sparse_tensor.indices().numpy()
        values = sparse_tensor.values().numpy()
        shape = sparse_tensor.shape

        # Write the data into a file
        x_group = write_file.create_group("X")
        x_group.create_dataset("indices", data=indices)
        x_group.create_dataset("values", data=values)
        x_group.attrs["shape"] = shape
