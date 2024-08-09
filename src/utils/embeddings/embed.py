# --- EMBEDDINGS ---


def get_data_embeddings(
    tensor_dataset: TensorDataset, model, batch_size: int = 1
) -> Path:
    r"""
    Computes the embeddings of the data using the model and saves them to a HDF5 file.

    Args:
        tensor_dataset (TensorDataset): The input tensor dataset.
        model: The model used to compute the embeddings.
        batch_size (int, optional): The batch size for the data loader. Defaults to 1.

    Returns:
        Path: The path to the saved HDF5 file containing the embeddings.
    """
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    # Open an HDF5 file
    with h5py.File(
        EMBEDDINGS_PATH / f"{model.cfg.model.model_name}-embeddings.h5", "w"
    ) as h5f:
        # Create a resizable dataset to store embeddings
        max_shape = (
            None,
            model.cfg.model.latent_dim,
        )  # None indicates that the dimension is resizable
        dset = h5f.create_dataset(
            "embeddings",
            shape=(0, model.cfg.model.latent_dim),
            maxshape=max_shape,
            dtype="float32",
        )

        for i, x in enumerate(dataloader):
            x = x[0]
            x.to(model.device)
            with torch.no_grad():
                encoded = model.encode(x)

            # Resize the dataset to accommodate new embeddings
            dset.resize(dset.shape[0] + encoded.shape[0], axis=0)
            # Write the new embeddings
            dset[-encoded.shape[0] :] = encoded.detach().cpu().numpy()

    return EMBEDDINGS_PATH / f"{model.cfg.model.model_name}-embeddings.h5"


def get_data_embeddings_transformer_version(
    tensor_dataset: TensorDataset, model, batch_size: int = 1
) -> Path:
    r"""
    Computes the embeddings of the data using the model and saves them to an HDF5 file.

    Args:
        tensor_dataset (TensorDataset): The input tensor dataset.
        model: The model used to compute the embeddings.
        batch_size (int, optional): The batch size for the data loader. Defaults to 1.

    Returns:
        Path: The path to the saved HDF5 file containing the embeddings.
    """
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    # Open an HDF5 file
    with h5py.File(
        EMBEDDINGS_PATH / f"{model.cfg.model.model_name}-embeddings.h5", "w"
    ) as h5f:
        # Create a resizable dataset to store embeddings
        max_shape = (
            None,
            model.cfg.model.latent_dim,
        )  # None indicates that the dimension is resizable
        dset = h5f.create_dataset(
            "mu_embeddings",
            shape=(0, model.cfg.model.latent_dim),
            maxshape=max_shape,
            dtype="float32",
        )

        for i, x in enumerate(dataloader):
            x = x[0]
            encoder_input_gene_idxs, aux_gene_idxs = model.gene_choice_module.choose(x)
            encoder_input_exprs_lvls = torch.gather(
                x, dim=1, index=encoder_input_gene_idxs
            )
            model.to(encoder_input_exprs_lvls.device)
            with torch.no_grad():
                _, mu, _ = model.encode(
                    encoder_input_gene_idxs, encoder_input_exprs_lvls
                )

            # Resize the dataset to accommodate new embeddings
            dset.resize(dset.shape[0] + mu.shape[0], axis=0)
            # Write the new embeddings
            dset[-mu.shape[0] :] = mu.detach().cpu().numpy()

    return EMBEDDINGS_PATH / f"{model.cfg.model.model_name}-embeddings.h5"
