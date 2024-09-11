import matplotlib
import matplotlib.pyplot as plt
import torch
from functools import partial
from typing import Callable, Dict, Any, List, Tuple
from utils.common_types import Batch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import seaborn as sns

# import scienceplots
# plt.style.use("science")


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


def plot_2d_latent(
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

            condition_value = condition_value.to("cpu").detach().numpy()
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


def plot_latent_tsne(
    processed_data: Batch,
    data_name: str,
    condition_value_name: str,
    condition_value_idxs: list,
    are_conditions_categorical: list,
    filename_comp: str,
    plot_dims: Tuple[int] = (0, 1),
    figsize: Tuple[float, float] = (6, 6),
    n_components: int = 2,
) -> List[matplotlib.figure.Figure]:

    figs_axs = [
        plt.subplots(figsize=figsize) for _ in condition_value_idxs
    ]  # Create a new figure and axes

    plot_dim_1, plot_dim_2 = plot_dims

    embedding = processed_data[data_name].to("cpu").detach().numpy()

    reducer = TSNE(n_components=n_components)
    embedding = reducer.fit_transform(embedding)

    for condition_value_idx, is_condition_categorical, (fig, ax) in zip(
        condition_value_idxs, are_conditions_categorical, figs_axs
    ):
        condition_value = processed_data[condition_value_name][:, condition_value_idx]
        condition_value = (
            condition_value.to(dtype=torch.long)
            if is_condition_categorical
            else condition_value
        )
        condition_value = condition_value.to("cpu").detach().numpy()
        scatter = ax.scatter(
            embedding[:, plot_dim_1],
            embedding[:, plot_dim_2],
            c=condition_value,
            cmap="tab10",
            s=1,
        )
        fig.colorbar(scatter, ax=ax)  # Add a colorbar to the figure
        ax.set_title(
            f"Validation set embedding | Condition id {condition_value_idx}"
        )  # Set the title of the plot
        plt.close(fig)

    figs = {
        f"{filename_comp}-{condition_value_idx}": fig_ax[0]
        for condition_value_idx, fig_ax in zip(condition_value_idxs, figs_axs)
    }

    return figs  # Return the figure object


def helper_make_barplot(
    x: List[str],
    y: List[float],
    title: str,
    xlab: str,
    ylab: str,
    rotation: int = 0,
    figsize: Tuple[float, float] = (10, 4),
) -> matplotlib.figure.Figure:
    r"""
    Make a barplot.

    Parameters:
    x (List[str]): The x-axis labels.
    y (List[float]): The y-axis values.
    title (str): The title of the plot.
    xlab (str): The label for the x-axis.
    ylab (str): The label for the y-axis.
    rotation (int): The rotation angle for x-axis labels.
    figsize (Tuple[float, float]): Size of the figure.

    Returns:
    matplotlib.figure.Figure: The figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, y, color="black")
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=rotation)

    plt.close(fig)

    return fig


def helper_plot_features_components_correlation(
    components: np.ndarray,
    feature_names: np.ndarray,
    correlation_threshold: float = 0,
    title: str = "",
    xlab: str = "",
    display_desc: bool = False,
    figsize: Tuple[float, float] = (10, 8),
) -> matplotlib.figure.Figure:
    r"""
    Plot the correlation of the features with the components.

    Parameters:
    components (np.ndarray): The PCA components.
    feature_names (np.ndarray): The names of the features.
    correlation_threshold (float): The threshold for displaying correlations.
    title (str): The title of the plot.
    xlab (str): The label for the x-axis.
    display_desc (bool): Whether to display the title.
    figsize (Tuple[float, float]): Size of the figure.

    Returns:
    matplotlib.figure.Figure: The figure object containing the plot.
    """
    df_disp = pd.DataFrame(
        data=components.T * 100, index=feature_names, columns=list(range(1, 11))
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        df_disp,
        annot=True,
        fmt=".0f",
        vmin=-100,
        vmax=100,
        linewidth=0.5,
        mask=abs(df_disp) < correlation_threshold,
        cmap=sns.diverging_palette(255, 10, as_cmap=True),
        ax=ax,
    )
    if display_desc:
        ax.set_title(title)
    ax.set_xlabel(xlab)

    plt.close(fig)

    return fig


def plot_gm_means_pca(
    batch: Batch,
    gm_means_name: str,
    filename_comp: str,
    n_components: int,
    barplot_figsize: Tuple[float, float] = (10, 4),
    correlation_threshold: float = 0,
) -> matplotlib.figure.Figure:
    r"""
    Perform PCA on gm_means and plot the explained variance ratio.

    Parameters:
    batch (Batch): The batch containing gm_means.
    gm_means_name (str): The key for gm_means in the batch.
    filename_comp (str): The filename component for saving the plot.
    n_components (int): Number of principal components to compute.
    figsize (Tuple[float, float]): Size of the figure.

    Returns:
    matplotlib.figure.Figure: The figure object containing the plot.
    """
    gm_means = batch[gm_means_name].detach().numpy()
    n_features = gm_means.shape[-1]

    # Fit and transform PCA
    pca = PCA(n_components=n_components)
    pca.fit(gm_means)

    # Plot explained variance ratio
    explained_var_ratio = helper_make_barplot(
        x=np.arange(1, n_components + 1),
        y=pca.explained_variance_ratio_,
        title="Explained Variance Ratio of Principal Components (Non Scaled)",
        xlab="Principal Component",
        ylab="Explained Variance Ratio",
        figsize=barplot_figsize,
    )

    # Plot component
    component_correlation = helper_plot_features_components_correlation(
        components=pca.components_,
        feature_names=np.arange(n_features),
        correlation_threshold=correlation_threshold,
    )

    return {
        f"{filename_comp}-expl_var_ratio": explained_var_ratio,
        f"{filename_comp}-comp_corr": component_correlation,
    }


def wrap_with_first_batch(func: Callable, **kwargs: Dict[str, Any]) -> Callable:
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
        return func(batch, **kwargs)

    return wrapped_function


def wrap_with_processed_dataset(
    func: Callable, n_batches: int, **kwargs: Dict[str, Any]
) -> Callable:

    def wrapped_function(
        processing_function: Callable,
        dataloader: torch.utils.data.DataLoader,
    ):
        dataloader_iterator = iter(dataloader)
        batch = next(dataloader_iterator)
        processed_data = processing_function(batch)

        for batch, batch_idx in zip(dataloader_iterator, range(n_batches)):
            batch = processing_function(batch=batch)

            for key in processed_data.keys():
                processed_data[key] = torch.cat(
                    [processed_data[key], batch[key]], dim=0
                )

        return func(processed_data, **kwargs)

    return wrapped_function


def wrap_with_dataloader(func: Callable, **kwargs: Dict[str, Any]) -> Callable:

    def wrapped_function(
        processing_function: Callable, dataloader: torch.utils.data.DataLoader
    ):
        return func(processing_function, dataloader, **kwargs)

    return wrapped_function


def wrap_with_just_processing_function_output(
    func: Callable, **kwargs: Dict[str, Any]
) -> Callable:

    def wrapped_function(
        processing_function: Callable, dataloader: torch.utils.data.Dataloader
    ):
        batch = processing_function({})
        return func(batch, **kwargs)

    return wrapped_function


_PLOTTING_FUNCTIONS: Dict[str, Callable] = {
    "latent_2d": partial(wrap_with_dataloader, func=plot_2d_latent),
    "latent_tsne": partial(wrap_with_processed_dataset, func=plot_latent_tsne),
    "gm_means_pca": partial(
        wrap_with_just_processing_function_output, plot_gm_means_pca
    ),
    "latent_pca": None,
    "patent_pca+tsne": None,
    "original_vs_reconstructed": partial(
        wrap_with_first_batch, func=plot_original_vs_reconstructed
    ),
    "images_with_conditions": partial(
        wrap_with_first_batch, func=plot_images_with_conditions
    ),
}


def get_plotting_function_taking_dataloader(
    plotting_func_name: str, **kwargs: Dict[str, Any]
) -> Callable:
    plotting_func: None | Callable = _PLOTTING_FUNCTIONS.get(plotting_func_name, None)

    if plotting_func is not None:
        return plotting_func(**kwargs)

    raise ValueError(
        f"The provided plotting_func_name {plotting_func_name} is invalid. Must be one of {list(_PLOTTING_FUNCTIONS.keys())}"
    )
