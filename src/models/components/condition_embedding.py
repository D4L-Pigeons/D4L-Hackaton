from re import L
from matplotlib import category
import torch
import torch.nn as nn
from argparse import Namespace
from src.utils.common_types import (
    Batch,
    ConfigStructure,
    format_structured_forward_output,
)
from src.utils.config import validate_config_structure
from typing import List
from einops import rearrange, repeat

# class ContinousValuedConditionEmbedding(nn.Module):
#     def __init__(self, cfg: Namespace) -> None:
#         self.token_embedding


class DiscreteValuedConditionEmbedding(nn.Module):
    r"""
    A module for embedding discrete-valued conditions and categories.

    Args:
        cfg (Namespace): The configuration object containing the following attributes:
            - n_conditions (int): The number of conditions.
            - n_categories_per_condition (List[int]): The number of categories per condition.
        embedding_dim (int): The dimension of the embedding vectors.

    Attributes:
        _condition_embeddings (nn.Embedding): The embedding layer for conditions.
        _category_embeddings (nn.Embedding): The embedding layer for categories.
        _cat_embed_start_per_cond (torch.LongTensor): The starting positions of category embeddings for each condition.

    Methods:
        embed(cond_ids, cat_ids):
            Embeds the given condition and category IDs.

    Raises:
        AssertionError: If the number of entries in n_categories_per_condition is not equal to n_conditions.
        AssertionError: If any condition has a non-positive number of categories.
        AssertionError: If any category ID is less than 0.
    """

    _config_structure: ConfigStructure = {
        "n_conditions": int,
        "n_categories_per_condition": [int],
    }

    def __init__(self, cfg: Namespace, embedding_dim: int) -> None:
        super(DiscreteValuedConditionEmbedding, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        # Initialize condition embeddings.
        self._condition_embeddings: nn.Embedding = nn.Embedding(
            num_embeddings=cfg.n_conditions + 1,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        nn.init.xavier_uniform_(self._condition_embeddings.weight)

        # Initialize category embeddings.
        assert (
            len(cfg.n_categories_per_condition) == cfg.n_conditions
        ), "The number of entries in n_categories_per_condition must be equal to n_conditions."
        assert all(
            [n_cat > 0 for n_cat in cfg.n_categories_per_condition]
        ), "All conditions should have positive number of categories."
        # Pad token on 0 pos
        self._cat_embed_start_per_cond: torch.LongTensor = torch.tensor(
            [0, 1] + cfg.n_categories_per_condition, dtype=torch.long
        ).cumsum(dim=0)
        n_categories_total: int = self._cat_embed_start_per_cond[-1]
        self._category_embeddings: nn.Embedding = nn.Embedding(
            num_embeddings=n_categories_total,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        nn.init.xavier_uniform_(self._category_embeddings.weight)

    def embed(self, cond_ids: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        assert (cat_ids >= 0).all(), "Some cat_ids are less than 0."
        cat_ids: torch.LongTensor = self._cat_embed_start_per_cond[
            cond_ids
        ] + cat_ids.to(torch.long)
        assert torch.isnan(cat_ids).any().logical_not()
        assert torch.isnan(self._condition_embeddings(cond_ids)).any().logical_not()
        assert torch.isnan(self._category_embeddings(cat_ids)).any().logical_not()
        condition_embeddings = self._condition_embeddings(cond_ids)
        category_embeddings = self._category_embeddings(cat_ids)
        # print(f"condition_embeddings.max() = {condition_embeddings.max()}")
        # print(f"category_embeddings.max() = {category_embeddings.max()}")
        output = condition_embeddings + category_embeddings
        # print(f"(condition_embeddings + category_embeddings).max() = {output.max()}")
        return output


class ConditionEmbeddingTransformer(nn.Module):
    r"""
    Condition Embedding Transformer module.

    Args:
        cfg (Namespace): Configuration object.

    Attributes:
        _cls_pooling_token (nn.Parameter): Parameter for the CLS pooling token.
        _condition_embedding_module (DiscreteValuedConditionEmbedding): Condition embedding module.
        _transformer_encoder (nn.TransformerEncoder): Transformer encoder.

    Methods:
        forward(batch: Batch) -> Batch:
            Forward pass of the module.

    """

    _config_structure: ConfigStructure = {
        "embedding_dim": int,
        "transformer": {"encoder_kwargs": Namespace, "encoder_layer_kwargs": Namespace},
        "embedding_module": Namespace,
    }

    def __init__(self, cfg: Namespace) -> None:
        super(ConditionEmbeddingTransformer, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._cls_pooling_token = nn.Parameter(
            torch.empty(size=(1, 1, cfg.embedding_dim))
        )
        torch.nn.init.xavier_normal_(self._cls_pooling_token)

        self._condition_embedding_module = DiscreteValuedConditionEmbedding(
            cfg=cfg.embedding_module, embedding_dim=cfg.embedding_dim
        )

        self._transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=cfg.embedding_dim,
                batch_first=True,
                **vars(cfg.transformer.encoder_layer_kwargs),
            ),
            **vars(cfg.transformer.encoder_kwargs),
        )

    def forward(
        self, batch: Batch, cond_ids_name: str, cat_ids_name: str, cond_embed_name: str
    ) -> Batch:

        # Embedding conditions
        condition_embeddings = self._condition_embedding_module.embed(
            cond_ids=batch[cond_ids_name], cat_ids=batch[cat_ids_name]
        )

        batch_size: int = condition_embeddings.shape[0]
        repeated_cls_pool_token: torch.Tensor = repeat(
            self._cls_pooling_token, "1 1 dim -> batch 1 dim", batch=batch_size
        )

        # Add cls pooling token
        condition_embeddings = torch.cat(
            [
                repeated_cls_pool_token,
                condition_embeddings,
            ],
            dim=1,  # sequence dim
        )

        assert torch.isnan(condition_embeddings).any().logical_not()

        # Apply transformer
        transformer_output = self._transformer_encoder(condition_embeddings)
        assert torch.isnan(transformer_output).any().logical_not()

        # Extract the embedding of the pooling token
        batch[cond_embed_name] = transformer_output[:, 0, :]

        return format_structured_forward_output(batch=batch)
