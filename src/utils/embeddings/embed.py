from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset


def _save_embeddings_to_hdf5(
    dataloader,
    model,
    file_path: Path,
    latent_dim: int,
    dataset_name: str = "embeddings",
    encoder_type: str = "default",
):
    """
    Helper function to compute and save embeddings to HDF5.
    """
    model.eval()
    with h5py.File(file_path, "w") as h5f:
        max_shape = (None, latent_dim)  # None indicates that the dimension is resizable
        dset = h5f.create_dataset(
            dataset_name,
            shape=(0, latent_dim),
            maxshape=max_shape,
            dtype="float32",
        )

        for x in dataloader:
            x = x[0].to(model.device)
            with torch.no_grad():
                if encoder_type == "transformer":
                    encoder_input_gene_idxs, aux_gene_idxs = (
                        model.gene_choice_module.choose(x)
                    )
                    encoder_input_exprs_lvls = torch.gather(
                        x, dim=1, index=encoder_input_gene_idxs
                    )
                    model.to(encoder_input_exprs_lvls.device)
                    _, embeddings, _ = model.encode(
                        encoder_input_gene_idxs, encoder_input_exprs_lvls
                    )
                else:
                    embeddings = model.encode(x)

            dset.resize(dset.shape[0] + embeddings.shape[0], axis=0)
            dset[-embeddings.shape[0] :] = embeddings.detach().cpu().numpy()


def get_data_embeddings(
    tensor_dataset: TensorDataset, model, batch_size: int = 1, file_path: Path = None
) -> Path:
    r"""
    Computes the embeddings of the data using the model and saves them to an HDF5 file.
    Args:
        tensor_dataset (TensorDataset): The input tensor dataset.
        model: The model used to compute the embeddings.
        batch_size (int, optional): The batch size for the data loader. Defaults to 1.
        file_path (Path, optional): Path to save the HDF5 file. Defaults to None, which generates a path based on the model name.
    Returns:
        Path: The path to the saved HDF5 file containing the embeddings.
    """
    if file_path is None:
        file_path = EMBEDDINGS_PATH / f"{model.cfg.model.model_name}-embeddings.h5"

    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)

    _save_embeddings_to_hdf5(dataloader, model, file_path, model.cfg.model.latent_dim)

    return file_path


def get_data_embeddings_transformer_version(
    tensor_dataset: TensorDataset, model, batch_size: int = 1, file_path: Path = None
) -> Path:
    r"""
    Computes the embeddings of the data using the transformer-based model and saves them to an HDF5 file.
    Args:
        tensor_dataset (TensorDataset): The input tensor dataset.
        model: The transformer-based model used to compute the embeddings.
        batch_size (int, optional): The batch size for the data loader. Defaults to 1.
        file_path (Path, optional): Path to save the HDF5 file. Defaults to None, which generates a path based on the model name.
    Returns:
        Path: The path to the saved HDF5 file containing the embeddings.
    """
    if file_path is None:
        file_path = EMBEDDINGS_PATH / f"{model.cfg.model.model_name}-embeddings.h5"

    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)

    _save_embeddings_to_hdf5(
        dataloader,
        model,
        file_path,
        model.cfg.model.latent_dim,
        dataset_name="mu_embeddings",
        encoder_type="transformer",
    )

    return file_path
