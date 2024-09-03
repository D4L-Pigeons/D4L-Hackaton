import matplotlib
import matplotlib.pyplot as plt
import torch
import neptune
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from functools import partial
import inspect
from typing import Callable, Dict, Any, List, Tuple
from src.utils.common_types import Batch


def plot_original_vs_reconstructed(
    batch: Any,  # Assuming batch is a dictionary-like object
    org_imgs_name: str,
    reconstructed_imgs_name: str,
    num_images: int,
    filename_comp: str,
    num_rows: int = 1,
    wspace: float = 0.5,
    hspace: float = 0.5,
    disp_img_size: int = 2,
) -> Dict[str, plt.Figure]:
    r"""
    Plots a grid of original and reconstructed images in an interleaved manner.

    Parameters:
    batch (Any): The batch containing original and reconstructed images.
    org_imgs_name (str): The key for original images in the batch.
    reconstructed_imgs_name (str): The key for reconstructed images in the batch.
    num_images (int): The number of images to display.
    num_rows (int): The number of rows to display the images.
    wspace (float): The amount of width reserved for blank space between subplots.
    hspace (float): The amount of height reserved for blank space between subplots.
    disp_img_size (int): The size of each displayed image.
    """

    org_imgs = batch[org_imgs_name]
    reconstructed_imgs = batch[reconstructed_imgs_name]

    # Calculate the number of columns needed
    num_cols = (
        num_images + num_rows - 1
    ) // num_rows  # Round up division for number of columns

    # Create a figure with a grid of subplots, interleaving original and reconstructed rows
    fig, axes = plt.subplots(
        num_rows * 2,
        num_cols,
        figsize=(num_cols * disp_img_size, num_rows * 2 * disp_img_size),
    )

    # Adjust the spacing between images
    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    # Plot original and reconstructed images in an interleaved manner
    for i in range(num_images):
        row_idx = i // num_cols  # Determine which row we are plotting in
        col_idx = i % num_cols  # Determine which column we are plotting in

        # Original images (on even-indexed rows)
        original = org_imgs[i][0]
        axes[2 * row_idx, col_idx].imshow(original, cmap="gray")
        axes[2 * row_idx, col_idx].axis("off")

        # Reconstructed images (on odd-indexed rows)
        reconstruction = reconstructed_imgs[i][0].detach()
        axes[2 * row_idx + 1, col_idx].imshow(reconstruction, cmap="gray")
        axes[2 * row_idx + 1, col_idx].axis("off")

    # Set titles next to the first image in each row
    for row_idx in range(num_rows):
        axes[2 * row_idx, 0].text(
            -0.2,
            0.5,
            "Original",
            va="center",
            ha="right",
            rotation=90,
            fontsize=10,
            transform=axes[2 * row_idx, 0].transAxes,
        )
        axes[2 * row_idx + 1, 0].text(
            -0.2,
            0.5,
            "Reconstructed",
            va="center",
            ha="right",
            rotation=90,
            fontsize=10,
            transform=axes[2 * row_idx + 1, 0].transAxes,
        )

    # fig.suptitle(f"Epoch: {epoch}")  # Add title with epoch number

    # Close the figure to release memory after logging
    plt.close(fig)

    return {filename_comp: fig}


def plot_images_with_conditions(
    batch: Batch,
    imgs_name: str,
    conditions_name: str,
    condition_values_name: str,
    disp_batch_size: int,
    disp_n_latent_samples: int,
    filename_comp: str,
    wspace: float = 0.5,
    hspace: float = 0.5,
    disp_img_size: int = 2,
    y_title_shift: float = 0.95,
) -> matplotlib.figure.Figure:
    r"""
    Display a grid of images with conditions and condition values as titles.

    Parameters:
    images (torch.Tensor): Tensor of images with shape (batch, n_latent_samples, 1, 28, 28).
    conditions (torch.Tensor): Tensor of conditions with shape (batch, n_conds).
    condition_values (torch.Tensor): Tensor of condition values with shape (batch, n_conds).
    disp_batch_size (int): Number of batches to display.
    disp_n_latent_samples (int): Number of latent samples to display.
    """
    imgs = batch[imgs_name]
    conditions = batch[conditions_name]
    condition_values = batch[condition_values_name]

    # There is no n_latent samples dim create dummy n_latent_samples dim.
    if len(imgs.shape) == 4:
        imgs.unsqueeze_(1)

    batch_size, n_latent_samples, _, image_height, image_width = imgs.shape

    batch_size = min(batch_size, disp_batch_size)
    n_latent_samples = min(n_latent_samples, disp_n_latent_samples)

    fig, axes = plt.subplots(
        batch_size,
        n_latent_samples,
        figsize=(n_latent_samples * disp_img_size, batch_size * disp_img_size),
    )

    # Adjust the spacing between images
    fig.subplots_adjust(wspace=wspace, hspace=hspace)  # Adjust these values as needed

    if batch_size == 1:
        axes = [axes]
    if n_latent_samples == 1:
        axes = [[ax] for ax in axes]

    for i in range(batch_size):
        for j in range(n_latent_samples):
            img = imgs[i, j, 0].cpu().numpy()
            axes[i][j].imshow(img, cmap="gray")
            axes[i][j].axis("off")

            # Set title for each image
            condition_text = f"Conditions: {conditions[i].cpu().numpy()}"
            condition_values_text = f"Values: {condition_values[i].cpu().numpy()}"
            title_text = f"{condition_text}\n{condition_values_text}"
            axes[i][j].set_title(title_text, fontsize=8)

    # fig.suptitle(f"Epoch: {epoch}", y=y_title_shift)  # Add title with epoch number
    plt.close(fig)

    return {filename_comp: fig}


def wrap_with_first_batch(func: Callable, *args, **kwargs) -> Callable:
    r"""
    A decorator that wraps a function to automatically pass the first batch
    from a dataloader as the first argument named 'batch'.

    Parameters:
    func (Callable): The function to be wrapped.

    Returns:
    Callable: The wrapped function.
    """

    def wrapped_function(
        processing_function: Callable,
        dataloader: torch.utils.data.DataLoader,
    ):
        batch = next(iter(dataloader))
        batch = processing_function(batch=batch)
        return func(batch, *args, **kwargs)

    return wrapped_function


def plot_latent(
    processing_function: Callable,
    dataloader: torch.utils.data.DataLoader,
    data_name: str,
    condition_value_name: str,
    condition_value_idxs: list,
    are_conditions_categorical: list,
    filename_comp: str,
    num_batches: None | int = None,
    plot_dims: Tuple[int] = (0, 1),
    figsize: Tuple[float, float] = (6, 6),
) -> List[matplotlib.figure.Figure]:

    figs_axs = [
        plt.subplots(figsize=figsize) for _ in condition_value_idxs
    ]  # Create a new figure and axes

    # fig, ax = plt.subplots(figsize=figsize)  # Create a new figure and axes

    if num_batches is None:
        num_batches = len(dataloader)

    plot_dim_1, plot_dim_2 = plot_dims

    colorbar_set = False

    for _, batch in zip(range(num_batches), dataloader):
        batch = processing_function(batch)
        embedding = batch[data_name]
        embedding = embedding.to("cpu").detach().numpy()
        for condition_value_idx, is_condition_categorical, (fig, ax) in zip(
            condition_value_idxs, are_conditions_categorical, figs_axs
        ):
            condition_value = batch[condition_value_name][:, condition_value_idx]
            condition_value = (
                condition_value.to(dtype=torch.long)
                if is_condition_categorical
                else condition_value
            )
            scatter = ax.scatter(
                embedding[:, plot_dim_1],
                embedding[:, plot_dim_2],
                c=condition_value,
                cmap="tab10",
                s=1,
            )
            if not colorbar_set:
                fig.colorbar(scatter, ax=ax)  # Add a colorbar to the figure
        colorbar_set = True

    for condition_value_idx, (fig, ax) in zip(condition_value_idxs, figs_axs):
        ax.set_title(
            f"Validation set embedding | Condition id {condition_value_idx}"
        )  # Set the title of the plot
        plt.close(fig)

    figs = {
        f"{filename_comp}-{condition_value_idx}": fig_ax[0]
        for condition_value_idx, fig_ax in zip(condition_value_idxs, figs_axs)
    }

    return figs  # Return the figure object


class NeptunePlotLogCallback(Callback):

    def __init__(
        self,
        plotting_function_taking_dataloader: Callable,
        command_name: str,
        neptune_plot_log_path: str,
        plot_file_base_name: str,
        command_dynamic_kwargs: Dict[str, Any] = {},
    ) -> None:
        self._plotting_function = plotting_function_taking_dataloader
        self._command_name = command_name
        self._command_dynamic_kwargs = command_dynamic_kwargs
        self._plot_file_base_name = plot_file_base_name
        self._neptune_plot_log_path = neptune_plot_log_path

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        val_dataloader = trainer.val_dataloaders
        run_processing_command_method = getattr(
            pl_module, "run_processing_command", None
        )

        assert (
            run_processing_command_method is not None
        ), "Provided pl_module has no attribute named run_processing_command."
        assert inspect.ismethod(run_processing_command_method) or inspect.isfunction(
            run_processing_command_method
        ), f"Attribute run_processing_command of pl_module is not a method."

        partial_run_processing_command_method = partial(
            run_processing_command_method,
            command_name=self._command_name,
            dynamic_kwargs=self._command_dynamic_kwargs,
            reset_loss_manager=True,
        )

        pl_module.train(False)

        figs = self._plotting_function(
            processing_function=partial_run_processing_command_method,
            dataloader=val_dataloader,
        )

        for fig_filename_end, fig in figs.items():
            # Log the plot to Neptune
            if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
                trainer.logger.experiment[
                    f"{self._neptune_plot_log_path}/{self._plot_file_base_name}-{fig_filename_end}-{trainer.current_epoch}"
                ].upload(neptune.types.File.as_image(fig))

            # Close the figure to release memory
            plt.close(fig)
