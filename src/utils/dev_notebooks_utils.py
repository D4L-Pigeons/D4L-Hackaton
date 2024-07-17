import torch
from torch import Tensor
import pandas as pd
from pandas import DataFrame
import anndata as ad
import scanpy as sc
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from utils import data_utils
from argparse import Namespace
from torch.utils.data import TensorDataset, DataLoader


# Note: only preprosessing, which keeps 0 as 0 is allowed if we want 0 to mean no expression
def prepare_data_naive_mixing_version(
    val_frac: float,
    divide_by_nonzero_median: bool,
    df_columns: List[str],
    random_state: int = 0,
    return_raw: bool = False,
) -> (
    Tuple[Tuple[Tensor, Tensor], Tuple[DataFrame, DataFrame], Tuple[Tensor, Tensor]]
    | Tuple[Tuple[Tensor, Tensor], Tuple[DataFrame, DataFrame]]
):
    data = data_utils.load_anndata(mode="train", plus_iid_holdout=True, normalize=False)
    sc.pp.log1p(data)
    data = data[data.obs["is_train"].apply(lambda x: x in ["train", "iid_holdout"])]

    tensor_data = torch.tensor(data.X.toarray())
    raw = torch.tensor(data.layers["counts"].toarray())
    df = pd.DataFrame(data.obs[df_columns])
    del data
    if divide_by_nonzero_median:
        nonzero_median = tensor_data[tensor_data > 0].median(dim=0, keepdim=True).values
        tensor_data = tensor_data / nonzero_median
        del nonzero_median

    train_tensor_data, val_tensor_data, train_df, val_df, train_raw, val_raw = (
        train_test_split(
            tensor_data, df, raw, test_size=val_frac, random_state=random_state
        )
    )

    if return_raw:
        return (
            (train_tensor_data, val_tensor_data),
            (train_df, val_df),
            (train_raw, val_raw),
        )

    return (train_tensor_data, val_tensor_data), (train_df, val_df)


def draw_umaps_pca(
    data_tensor: Tensor, df: DataFrame, n_comps: int = 20, n_neighbors: int = 10
) -> None:
    r"""
    Computes pca and draws umaps for each column in df.
    """
    assert (
        n_comps <= data_tensor.shape[1]
    ), "n_comps cannot be greater than the number of features"
    ad_tmp = ad.AnnData(X=data_tensor.numpy(), obs=df)
    sc.pp.pca(ad_tmp, n_comps=n_comps)
    sc.pp.neighbors(ad_tmp, n_neighbors=n_neighbors, use_rep="X_pca")
    sc.tl.umap(ad_tmp)

    for col_name in df.columns:
        sc.pl.umap(ad_tmp, color=col_name)

    for col_name in df.columns:
        sc.pl.pca(ad_tmp, color=col_name)


def dict_to_namespace(d):
    r"""
    Recursively converts a dictionary into a Namespace object.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)
    return Namespace(**d)


def get_data_embeddings_transformer_version(
    tensor_dataset: TensorDataset, model
) -> Tuple[Tensor, Tensor]:

    dataloader = DataLoader(tensor_dataset, batch_size=1, shuffle=False)
    # sampled_embeddings = []
    mu_embeddings = torch.empty(0, model.cfg.model.latent_dim)
    for x in dataloader:
        x = x[0]
        encoder_input_gene_idxs, aux_gene_idxs = model.gene_choice_module.choose(x)
        encoder_input_exprs_lvls = torch.gather(x, dim=1, index=encoder_input_gene_idxs)
        # print(encoder_input_gene_idxs.device, encoder_input_exprs_lvls.device)
        # model.to("cpu")
        model.to(encoder_input_exprs_lvls.device)
        z, mu, _ = model.encode(encoder_input_gene_idxs, encoder_input_exprs_lvls)
        # sampled_embeddings.append(z)
        mu_embeddings = torch.cat([mu_embeddings, mu], dim=0)
    # sampled_embeddings = torch.cat(sampled_embeddings, dim=0)
    # return sampled_embeddings, mu_embeddings\
    return mu_embeddings
