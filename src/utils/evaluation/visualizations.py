from typing import List

import anndata as ad
import scanpy as sc
from numpy import ndarray
from pandas import DataFrame


def draw_umaps_pca(
    data: ndarray,
    df: DataFrame,
    n_comps: int = 20,
    n_neighbors: int = 10,
    plot_types: List[str] = ["umap", "pca"],
) -> None:
    r"""
    Computes PCA and draws UMAPs for each column in the DataFrame.

    Parameters:
        data (ndarray): The input data array.
        df (DataFrame): The DataFrame containing the columns to visualize.
        n_comps (int, optional): The number of principal components to compute in PCA. Defaults to 20.
        n_neighbors (int, optional): The number of neighbors to consider in UMAP. Defaults to 10.
        plot_types (List[str]): A list specifying which plots to generate ("umap", "pca"). Defaults to both.
    """
    assert (
        n_comps <= data.shape[1]
    ), "n_comps cannot be greater than the number of features"

    ad_tmp = ad.AnnData(X=data, obs=df)

    sc.pp.pca(ad_tmp, n_comps=n_comps)
    sc.pp.neighbors(ad_tmp, n_neighbors=n_neighbors, use_rep="X_pca")
    sc.tl.umap(ad_tmp)

    # Computing UMAP
    if "umap" in plot_types:
        sc.tl.umap(ad_tmp)
        for col_name in df.columns:
            sc.pl.umap(ad_tmp, color=col_name)

    # Generating PCA plots
    if "pca" in plot_types:
        for col_name in df.columns:
            sc.pl.pca(ad_tmp, color=col_name)
