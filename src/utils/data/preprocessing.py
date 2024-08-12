from argparse import Namespace
from typing import Callable, Dict, Generator, List, TypeAlias

import h5py
import torch
from scipy.sparse import csr_matrix
from tdigest import TDigest
from torch import Tensor
from tqdm import tqdm

from utils.paths import PREPROCESSED_DATA_PATH, RAW_DATA_PATH

TensorGenerator: TypeAlias = Generator[Tensor, None, None]


def _get_chunk_tensor_generator(
    X: h5py.Group, n_samples: int, chunk_size: int, n_features: int, desc=None
) -> TensorGenerator:
    r"""
    Generate chunks of data from an h5py.Group object.

    Args:
        X (h5py.Group): The h5py.Group object containing the data.
        n_samples (int): The total number of samples in the data.
        chunk_size (int): The size of each chunk.
        n_features (int): The number of features in each sample.
        desc (str, optional): Description for the progress bar. Defaults to None.

    Yields:
        Tensor: A chunk of data as a PyTorch tensor.

    """
    progress_bar = tqdm(
        range(0, n_samples, chunk_size),
        desc=desc,
    )
    for chunk_idx in progress_bar:
        chunk_indptr = X["indptr"][
            chunk_idx : (chunk_idx + chunk_size + 1)
        ]  # +1 because of using slice
        chunk_shape = (
            len(chunk_indptr) - 1,
            n_features,
        )  # The len(chunk_indptr) is always 1 + the len of the current chunk
        chunk_start, chunk_end = chunk_indptr[0], chunk_indptr[-1]
        chunk_data = X["data"][chunk_start:chunk_end]
        chunk_indices = X["indices"][chunk_start:chunk_end]
        chunk_tensor = torch.tensor(
            csr_matrix(
                (
                    chunk_data,
                    chunk_indices,
                    chunk_indptr - chunk_start,
                ),  # The chunk_start is subtracted to make chunk_indpts start with 0.
                shape=chunk_shape,
            ).toarray()
        )
        # print(f"_get_chunk_tensor_generator: {(chunk_tensor == 0).sum()}")
        yield chunk_tensor


def _estimate_nonzero_median(
    chunk_tensor_generator: TensorGenerator,
    n_features: int,
    delta: float = 0.01,
    K: int = 25,
) -> Tensor:
    r"""
    Estimate the median values for nonzero values in each column in a sequential manner.

    Args:
        chunk_tensor_generator (TensorGenerator): A generator that yields chunked tensors.
        n_features (int): The number of features (columns) in the tensors.
        delta (float, optional): The accuracy parameter for the t-digest algorithm. Defaults to 0.01.
        K (int, optional): The compression parameter for the t-digest algorithm. Defaults to 25.

    Returns:
        Tensor: A tensor containing the estimated median values for each column.
    """
    columns_tds = [TDigest(delta=delta, K=K) for _ in range(n_features)]
    # progress_bar = tqdm(chunk_tensor_generator, desc="Calculating columnwise medians.")
    for chunk_tensor in chunk_tensor_generator:  # progress_bar:
        # column_progress_bar = tqdm(enumerate(column#s_tds))
        for j, td in enumerate(columns_tds):  # column_progress_bar:
            nonzero_row_indices = torch.nonzero(chunk_tensor[:, j], as_tuple=True)[0]
            td.batch_update(values=chunk_tensor[nonzero_row_indices, j])

    return torch.tensor(
        [td.percentile(0.5) for td in columns_tds], dtype=torch.float32
    ).unsqueeze(
        0
    )  # Adding dummy dimension to create row tensor.


def _divide_by_nonzero_median_chunk_tensor_generator(
    chunk_tensor_generator: TensorGenerator, n_features: int, **kwargs: Dict
) -> TensorGenerator:
    r"""
    Divides each element of the input tensor `x` by the median of non-zero elements along each column.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Tensor with each element divided by the median of non-zero elements along each column.
    """
    print("Estimating median start")
    features_medians: Tensor = _estimate_nonzero_median(
        chunk_tensor_generator=chunk_tensor_generator, n_features=n_features
    )
    print("Estimating median end")
    for chunk_tensor in chunk_tensor_generator:
        # chunk_tensor = chunk_tensor / features_medians
        # print(
        #     f"_divide_by_nonzero_median_chunk_tensor_generator: {(chunk_tensor == 0).sum()}"
        # )
        yield chunk_tensor / features_medians


def _normalize_total_chunk_tensor_generator(
    chunk_tensor_generator: TensorGenerator, **kwargs: Dict
) -> TensorGenerator:
    r"""
    Normalize the total chunk tensor generator.

    This function takes a generator of chunk tensors and normalizes each chunk tensor by dividing it by the sum of its elements along the second dimension.

    Args:
        chunk_tensor_generator (TensorGenerator): A generator that yields chunk tensors.

    Yields:
        Tensor: The normalized chunk tensor.

    """
    for chunk_tensor in chunk_tensor_generator:
        # chunk_tensor = chunk_tensor / chunk_tensor.sum(dim=1, keepdim=True)
        # print(f"_normalize_total_chunk_tensor_generator: {(chunk_tensor == 0).sum()}")
        yield chunk_tensor / chunk_tensor.sum(dim=1, keepdim=True)


def _log1p_chunk_tensor_generator(
    chunk_tensor_generator: TensorGenerator, **kwargs: Dict
) -> TensorGenerator:
    r"""
    Applies the log1p function to each chunk tensor in the given chunk tensor generator.

    Args:
        chunk_tensor_generator (TensorGenerator): A generator that yields chunk tensors.

    Yields:
        Tensor: The chunk tensor with log1p applied.

    """
    for chunk_tensor in chunk_tensor_generator:
        # chunk_tensor = chunk_tensor.log1p()
        # print(f"_log1p_chunk_tensor_generator: {(chunk_tensor == 0).sum()}")
        yield chunk_tensor.log1p()


_TRANSFORMS_GENERATORS_DICT: Dict[str, Callable] = {
    "normalize_total": _normalize_total_chunk_tensor_generator,
    "divide_by_nonzero_median": _divide_by_nonzero_median_chunk_tensor_generator,
    "log1p": _log1p_chunk_tensor_generator,
}


def _apply_transforms_chunk_tensor_generator(
    chunk_tensor_generator: TensorGenerator, transforms: List[str], **kwargs: Dict
) -> TensorGenerator:
    r"""
    Applies a series of transforms to a chunk tensor generator.

    Args:
        chunk_tensor_generator (TensorGenerator): The input chunk tensor generator.
        transforms (List[str]): A list of transform keys specifying the transforms to apply.
        **kwargs (Dict): Additional keyword arguments to pass to the transform functions.

    Returns:
        TensorGenerator: The transformed chunk tensor generator.
    """
    for transform_key in transforms:
        chunk_tensor_generator = _TRANSFORMS_GENERATORS_DICT[transform_key](
            chunk_tensor_generator, **kwargs
        )

    return chunk_tensor_generator


def _to_sparse_chunk_tensor_generator(
    chunk_tensor_generator: TensorGenerator,
) -> TensorGenerator:
    for chunk_tensor in chunk_tensor_generator:
        yield chunk_tensor.to_sparse().values().numpy()


# def _substitute_transformed_values(
#     X: h5py.Group,
#     sparse_chunk_tensor_generator: TensorGenerator,
#     chunk_size: int,
#     n_samples: int,
# ) -> None:
#     r"""
#     Substitutes transformed values in the given h5py.Group object with the values from the sparse chunk tensor generator.

#     Args:
#         X (h5py.Group): The h5py.Group object containing the data.
#         sparse_chunk_tensor_generator (TensorGenerator): The generator that yields sparse chunk tensors.
#         chunk_size (int): The size of each chunk.

#     Returns:
#         None
#     """
#     # chunk_idx = 0
#     for chunk_idx, sparse_chunk_tensor in zip(
#         range(0, n_samples, chunk_size), sparse_chunk_tensor_generator
#     ):
#         chunk_start, chunk_end = X["indptr"][
#             [
#                 chunk_idx,
#                 min(chunk_idx + chunk_size, n_samples),
#             ]  # I do no like this min here. Iterating through two ranges could solve the issue, but idk if one sould care about it too much.
#         ]  # Not adding +1 as in slicing.
#         # print(type(X["data"][chunk_start:chunk_end]), type(sparse_chunk_tensor))
#         # print(X["data"][chunk_start:chunk_end].shape, sparse_chunk_tensor.shape)
#         X["data"][chunk_start:chunk_end] = sparse_chunk_tensor
#         # chunk_idx += chunk_size


def _substitute_transformed_values(
    X: h5py.Group,
    sparse_chunk_tensor_generator: TensorGenerator,
    chunk_size: int,
    n_samples: int,
) -> None:
    r"""
    Substitutes transformed values in the given h5py.Group object with the values from the sparse chunk tensor generator.

    Args:
        X (h5py.Group): The h5py.Group object containing the data.
        sparse_chunk_tensor_generator (TensorGenerator): The generator that yields sparse chunk tensors.
        chunk_size (int): The size of each chunk.

    Returns:
        None
    """
    chunk_indices = range(0, n_samples, chunk_size)
    for chunk_idx, sparse_chunk_tensor in zip(
        chunk_indices, sparse_chunk_tensor_generator
    ):
        chunk_start, chunk_end = X["indptr"][
            chunk_idx : min(chunk_idx + chunk_size, n_samples)
        ]
        X["data"][chunk_start:chunk_end] = sparse_chunk_tensor


def preprocess_and_save_dataset(cfg: Namespace) -> None:
    # Paths configuration and validation
    read_path = RAW_DATA_PATH / cfg.read_file.filename
    write_path = PREPROCESSED_DATA_PATH / cfg.write_file.filename

    assert read_path.exists(), f"File not found: '{read_path}'."
    if not write_path.exists():
        write_path.touch()

    # Processing
    with h5py.File(write_path, "w") as write_file:
        with h5py.File(read_path, mode="r") as read_file:
            # Copy groups to write file
            for key in read_file.keys():
                if key == "X" or key in cfg.groups_to_clone:
                    read_file.copy(key, write_file)

        n_samples, n_features = write_file["X"].attrs["shape"]
        assert (
            cfg.chunk_size < n_samples
        ), "chunk_size must be less than the number of samples."

        # Create a list of transformations to apply
        transformations = [
            lambda: _get_chunk_tensor_generator(
                X=write_file["X"],
                n_samples=n_samples,
                chunk_size=cfg.chunk_size,
                n_features=n_features,
                desc="Data preprocessing",
            )
        ]

        # Apply transforms in sequence
        for transform in cfg.transforms:
            transformations.append(
                lambda gen=transformations[-1]: _TRANSFORMS_GENERATORS_DICT[transform](
                    gen(), n_features=n_features
                )
            )

        # Convert the result to sparse format
        transformations.append(
            lambda gen=transformations[-1]: _to_sparse_chunk_tensor_generator(gen())
        )
        try:
            print("Starting data preprocessing...")
            # Substitute transformed values in the write file
            _substitute_transformed_values(
                X=write_file["X"],
                sparse_chunk_tensor_generator=transformations[-1](),
                chunk_size=cfg.chunk_size,
                n_samples=n_samples,
            )
            print("Data preprocessing completed successfully.")

        except Exception as e:
            print(f"An error occurred during preprocessing: {e}")
            raise


# def preprocess_and_save_dataset(
#     cfg: Namespace,
# ) -> None:
#     r"""
#     Preprocesses the data and saves it to an HDF5 file. Does processing in a sequential manner without loading the whole dataset at once.

#     Args:
#         cfg (Namespace): The configuration object containing preprocessing options.
#             - read_file (Namespace): The configuration object for the input file.
#                 - filename (str): The name of the input file.
#             - write_file (Namespace): The configuration object for the output file.
#                 - filename (str): The name of the output file.
#             - transforms (List[str]): A list defining the sequence of data preprocessing transformations.
#             - groups_to_clone (List[str]): A list of the groups to be cloned into the write file from the read file.
#             - chunk_size (int): Size of the chunk used in processing of the data.
#     Returns:
#         None
#     """
#     read_path = RAW_DATA_PATH / cfg.read_file.filename
#     assert (
#         read_path.exists()
#     ), f"The file poited to by a path='{read_path}' does not exist."
#     write_path = PREPROCESSED_DATA_PATH / cfg.write_file.filename
#     if not write_path.exists():
#         write_path.touch()

#     with h5py.File(write_path, "w") as write_file:
#         with h5py.File(read_path, mode="r") as read_file:
#             read_file.copy("X", write_file)
#             for key in read_file.keys():
#                 if key in cfg.groups_to_clone:
#                     read_file.copy(key, write_file)
#         n_samples, n_features = write_file["X"].attrs["shape"]
#         assert (
#             cfg.chunk_size < n_samples
#         ), "The chunk_size must be less than the number of samples."

#         # Generator yielding dense tensors in chunks of size chunk_size.
#         chunk_tensor_generator = _get_chunk_tensor_generator(
#             X=write_file["X"],
#             n_samples=n_samples,
#             chunk_size=cfg.chunk_size,
#             n_features=n_features,
#             desc="Data preprocessing",
#         )
#         # Generator yielding transformed tensors.
#         chunk_tensor_generator = _apply_transforms_chunk_tensor_generator(
#             chunk_tensor_generator=chunk_tensor_generator,
#             transforms=cfg.transforms,
#             n_features=n_features,
#         )
#         # Generator yielding nonzero values representing a tensor after transformations. Their indices are the same as befor transformations.
#         chunk_tensor_generator = _to_sparse_chunk_tensor_generator(
#             chunk_tensor_generator=chunk_tensor_generator
#         )
#         # Saving the transformed values in the write_file X group.
#         _substitute_transformed_values(
#             X=write_file["X"],
#             sparse_chunk_tensor_generator=chunk_tensor_generator,
#             chunk_size=cfg.chunk_size,
#             n_samples=n_samples,
#         )
#     shape = read_file["X"].attrs[
#         "shape"
#     ]  # Assuming the shape is stored as an attribute

# data = csr_matrix((data, indices, indptr), shape=shape)
# # Convert the sparse matrix to a PyTorch tensor
# data = torch.tensor(data.toarray())

# # # Apply transformations in specified order
# # for transform_key in cfg.transforms:
# #     tensor_data = _transform_tensor(transform_key, tensor_data)

# # Go back to sparse format
# data = data.to_sparse()
# indices = data.indices().numpy()
# values = data.values().numpy()
# shape = data.shape

# # Write the data into a file
# # x_group = write_file.create_group("X")
# # x_group.create_dataset("indices", data=indices)
# # x_group.create_dataset("values", data=values)
# # x_group.attrs["shape"] = shape
