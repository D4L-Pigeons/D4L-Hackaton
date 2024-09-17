import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd


# try:
#     import scienceplots

#     plt.style.use("science")
# except:
#     pass


def barplot(
    height: np.ndarray,
    x_ticks: list,
    title: None | str = None,
    xlab: None | str = None,
    ylab: None | str = None,
    rotation: int = 0,
    figsize: Tuple[float, float] = (10, 4),
) -> plt.Figure:
    r"""
    Create a bar plot with the given parameters.

    Parameters:
        height (np.ndarray): Array of heights for the bars.
        x_ticks (list): Array of labels for the x-axis ticks.
        title (str): Title of the plot.
        xlab (str): Label for the x-axis.
        ylab (str): Label for the y-axis.
        rotation (int, optional): Rotation angle for x-axis tick labels. Default is 0.
        figsize (Tuple[float, float], optional): Size of the figure. Default is (10, 4).

    Returns:
        plt.Figure: The created bar plot figure.
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x=x_ticks, height=height, color="black")
    ax.set_xticks(range(len(x_ticks)))
    ax.set_xticklabels(x_ticks, rotation=rotation)
    if title is not None:
        ax.set(title=title)
    if xlab is not None:
        ax.set(xlabel=xlab)
    if xlab is not None:
        ax.set(ylabel=ylab)

    plt.close(fig)

    return fig


def features_components_corrplot(
    components: np.ndarray,
    feature_names: None | List[str] = None,
    correlation_threshold: float = 0,
    title: str = "",
    xlab: str = "",
    display_desc: bool = False,
    figsize: Tuple[float, float] = (10, 8),
) -> plt.Figure:
    r"""
    Plot the correlation of the features with the components.

    Parameters:
    components (np.ndarray): The PCA components.
    feature_names (None | List[str]): The names of the features.
    correlation_threshold (float): The threshold for displaying correlations.
    title (str): The title of the plot.
    xlab (str): The label for the x-axis.
    display_desc (bool): Whether to display the title.
    figsize (Tuple[float, float]): Size of the figure.

    Returns:
    matplotlib.figure.Figure: The figure object containing the plot.
    """

    if feature_names is None:
        feature_names = list(range(components.T.shape[0]))

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


class ColoredScatterplot:
    r"""
    A class to create and manage a colored scatter plot.

    Attributes:
    -----------
    _x_dim : int
        The dimension of the data to be used for the x-axis.
    _y_dim : int
        The dimension of the data to be used for the y-axis.
    _fig : plt.Figure
        The figure object for the plot.
    _ax : plt.Axes
        The axes object for the plot.
    _colorbar_set : bool
        A flag indicating whether the colorbar has been set.
    _cmap : matplotlib.colors.Colormap
        The colormap to be used for the scatter plot.

    Methods:
    --------
    __init__(n_colors: None | int = None, x_dim: int = 0, y_dim: int = 1, figsize: Tuple[float, float] = (6, 6)):
        Initializes the ColoredScatterplot with optional parameters for number of colors, x and y dimensions, and figure size.

    add_points(data: np.ndarray, colors: np.ndarray) -> None:
        Adds points to the scatter plot with the given data and colors.

    fig() -> plt.Figure:
        Returns the figure object for the plot.

    __call__(data: np.ndarray, colors: np.ndarray) -> plt.Figure:
        Adds points to the scatter plot and returns the figure object.
    """

    def __init__(
        self,
        n_colors: None | int = None,
        x_dim: int = 0,
        y_dim: int = 1,
        figsize: Tuple[float, float] = (6, 6),
    ):

        self._x_dim: int = x_dim
        self._y_dim: int = y_dim

        fig, ax = plt.subplots(figsize)
        self._fig: plt.Figure = fig
        self._ax: plt.Axes = ax

        self._colorbar_set: bool = False
        self._cmap = plt.get_cmap(name="tab10", lut=n_colors)

    def add_points(self, data: np.ndarray, colors: np.ndarray) -> None:
        x, y = data[:, self._x_dim], data[:, self._y_dim]
        scatter = self._ax.scatter(x=x, y=y, s=1, c=colors, cmap=self._cmap)

        if not self._colorbar_set:
            self._fig.colorbar(mappable=scatter, cax=self._ax)
            self._colorbar_set = True

    @property
    def fig(self) -> plt.Figure:
        return self._fig

    def __call__(self, data: np.ndarray, colors: np.ndarray) -> plt.Figure:
        self.add_points(data=data, colors=colors)

        return self.fig


def original_vs_reconstructed_plot(
    org_imgs: np.ndarray,
    reconstructed_imgs: np.ndarray,
    num_images: int,
    num_rows: int = 1,
    wspace: float = 0.5,
    hspace: float = 0.5,
    disp_img_size: int = 2,
) -> plt.Figure:
    r"""
    Plots original images alongside their reconstructed counterparts in a grid layout.

    Args:
        org_imgs (np.ndarray): Array of original images.
        reconstructed_imgs (np.ndarray): Array of reconstructed images.
        num_images (int): Number of images to display.
        num_rows (int, optional): Number of rows in the grid. Defaults to 1.
        wspace (float, optional): Width space between images. Defaults to 0.5.
        hspace (float, optional): Height space between images. Defaults to 0.5.
        disp_img_size (int, optional): Display size of each image. Defaults to 2.

    Returns:
        plt.Figure: The matplotlib figure object containing the plot.
    """

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

    # Close the figure to release memory after logging
    plt.close(fig)

    return fig


def images_with_conditions_plot(
    imgs: np.ndarray,
    conditions: np.ndarray,
    condition_values: np.ndarray,
    disp_batch_size: int,
    disp_n_latent_samples: int,
    wspace: float = 0.5,
    hspace: float = 0.5,
    disp_img_size: int = 2,
    y_title_shift: float = 0.95,
) -> plt.Figure:
    r"""
    Plots a grid of images with associated conditions and condition values.

    Parameters:
        imgs (np.ndarray): Array of images with shape (batch_size, n_latent_samples, channels, height, width).
        conditions (np.ndarray): Array of conditions corresponding to each image.
        condition_values (np.ndarray): Array of condition values corresponding to each image.
        disp_batch_size (int): Number of images to display in the batch dimension.
        disp_n_latent_samples (int): Number of latent samples to display.
        wspace (float, optional): Width space between images in the grid. Default is 0.5.
        hspace (float, optional): Height space between images in the grid. Default is 0.5.
        disp_img_size (int, optional): Size of each displayed image. Default is 2.
        y_title_shift (float, optional): Shift for the y-axis title. Default is 0.95.

    Returns:
        plt.Figure: A dictionary with the filename component as the key and the figure as the value.
    """

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

    plt.close(fig)

    return fig


def pairplot(
    data: np.ndarray,
    condition_values: np.ndarray,
    hue_dim: int,
    n_data_cols: int,
    **kwargs: Dict[str, Any],
) -> plt.Figure:
    r"""
    Generates a pairplot for the given data with a specified hue dimension.

    Parameters:
    data (np.ndarray): The input data array.
    condition_values (np.ndarray): Array containing condition values for hue.
    hue_dim (int): The dimension index in condition_values to be used for hue.
    n_data_cols (int): Number of data columns to include in the pairplot.
    **kwargs (Dict[str, Any]): Additional keyword arguments to pass to seaborn.pairplot.

    Returns:
    plt.Figure: The matplotlib figure object containing the pairplot.
    """

    hue = condition_values[:, [hue_dim]]
    df = pd.DataFrame(
        data=np.concatenate([data[:, :n_data_cols], hue], axis=-1),
        columns=list(range(n_data_cols)) + ["hue"],
    )
    unique_hue_values = pd.Series(data=hue[:, 0]).unique()
    custom_palette = {
        value: color
        for value, color in zip(
            unique_hue_values, sns.color_palette("tab10", len(unique_hue_values))
        )
    }
    sns_plot = sns.pairplot(df, hue="hue", **kwargs, palette=custom_palette)
    fig = sns_plot.figure

    plt.close(fig)

    return fig
