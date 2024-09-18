r"""
A module for embedding conditions and categories.

This module provides classes for embedding conditions and categories in various ways. It includes classes for embedding continuous-valued conditions, continuous vector conditions, discrete-valued conditions, discrete vector conditions, and condition sets.

Classes:
- ContinousValuedConditionEmbedding: A module for embedding continuous-valued conditions.
- ContinousVectorConditioneEmbedding: A module for embedding continuous vector conditions.
- DiscreteValuedConditionEmbedding: A module for embedding discrete-valued conditions and categories.
- DiscreteVectorConditionEmbedding: A module for embedding discrete vector conditions.
- ConditionEmbeddingTransformer: A module for embedding conditions when some of the values may be dropped.
- ConditionSetEmbeddingTransformer: A module for embedding condition sets.

"""

import torch
import torch.nn as nn
from einops import repeat
from argparse import Namespace
from typing import Dict, Any

from utils.common_types import (
    Batch,
    ConfigStructure,
    StructuredForwardOutput,
    format_structured_forward_output,
)
from utils.config import validate_config_structure


class ContinousValuedConditionEmbedding(nn.Module):
    def __init__(self, cfg: Namespace) -> None:
        raise NotImplementedError


class ContinousVectorConditioneEmbedding(nn.Module):
    def __init__(self, cfg: Namespace) -> None:
        raise NotImplementedError


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

        assert (
            len(cfg.n_categories_per_condition) == cfg.n_conditions
        ), "The number of entries in n_categories_per_condition must be equal to n_conditions."
        assert all(
            [n_cat > 0 for n_cat in cfg.n_categories_per_condition]
        ), "All conditions should have positive number of categories."

        # Set the starting index of condition values corresponding to each condition. Pad token has position 0.
        self._cat_embed_start_per_cond: torch.LongTensor = torch.tensor(
            [0, 1] + cfg.n_categories_per_condition, dtype=torch.long
        ).cumsum(dim=0)

        # Initialize category embeddings.
        n_categories_total: int = self._cat_embed_start_per_cond[-1]
        self._category_embeddings: nn.Embedding = nn.Embedding(
            num_embeddings=n_categories_total,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        nn.init.xavier_uniform_(self._category_embeddings.weight)

    def embed(self, cond_ids: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        r"""
        Embeds the given condition and category IDs into a tensor.

        Args:
            cond_ids (torch.Tensor): The condition IDs.
            cat_ids (torch.Tensor): The category IDs.

        Returns:
            torch.Tensor: The embedded tensor.

        Raises:
            AssertionError: If any cat_ids are less than 0.
            AssertionError: If any cat_ids contain NaN values.
            AssertionError: If any condition embeddings contain NaN values.
            AssertionError: If any category embeddings contain NaN values.
        """

        assert (cat_ids >= 0).all(), "Some cat_ids are less than 0."

        # Map condition category values to appropriate category token emebddings.
        cat_ids: torch.LongTensor = self._cat_embed_start_per_cond[
            cond_ids
        ] + cat_ids.to(torch.long)

        assert (
            torch.isnan(cat_ids).any().logical_not()
        ), "Some cat_ids contain NaN values."
        assert (
            torch.isnan(self._condition_embeddings(cond_ids)).any().logical_not()
        ), "Some condition embeddings contain NaN values."
        assert (
            torch.isnan(self._category_embeddings(cat_ids)).any().logical_not()
        ), "Some category embeddings contain NaN values."

        # Obtain condition and category embedding and sum them to get condition & category embedding.
        condition_embeddings = self._condition_embeddings(cond_ids)
        category_embeddings = self._category_embeddings(cat_ids)
        output = condition_embeddings + category_embeddings

        return output


class DiscreteVectorConditionEmbedding(nn.Module):
    def __init__(self, cfg: Namespace) -> None:
        raise NotImplementedError


class ConditionEmbeddingTransformer(nn.Module):
    r"""A module for embedding conditions when some of the values may be dropped like in expression tables.
    Later stage of the project if even used.
    """

    def __init__(self, cfg: Namespace) -> None:
        raise NotImplementedError


class ConditionSetEmbeddingTransformer(nn.Module):
    r"""
    Condition Set Embedding Transformer module.

    Args:
        cfg (Namespace): Configuration object.

    Attributes:
        _cls_pooling_token (nn.Parameter): Parameter for the CLS pooling token.
        _condition_embedding_module (DiscreteValuedConditionEmbedding): Condition embedding module.
        _transformer_encoder (nn.TransformerEncoder): Transformer encoder.

    Methods:
        forward(batch: Batch) -> StructuredForwardOutput:
            Forward pass of the module.

    """

    _config_structure: ConfigStructure = {
        "embedding_dim": int,
        "transformer": {"encoder_kwargs": Namespace, "encoder_layer_kwargs": Namespace},
        "embedding_module": Namespace,
    }

    def __init__(self, cfg: Namespace) -> None:
        super(ConditionSetEmbeddingTransformer, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        # Initialize condition set reprezentaion token.
        self._cls_pooling_token = nn.Parameter(
            torch.empty(size=(1, 1, cfg.embedding_dim))
        )
        torch.nn.init.xavier_normal_(self._cls_pooling_token)

        # Initialize condition emebdding module.
        self._condition_embedding_module = DiscreteValuedConditionEmbedding(
            cfg=cfg.embedding_module, embedding_dim=cfg.embedding_dim
        )

        # Setup a transformer for condition set embedding.
        self._transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=cfg.embedding_dim,
                batch_first=True,
                **vars(cfg.transformer.encoder_layer_kwargs),
            ),
            **vars(cfg.transformer.encoder_kwargs),
        )

    @staticmethod
    def _parse_hparams_to_dict(cfg: Namespace) -> Dict[str, Any]:
        r"""
        Parse the hyperparameters to a dictionary for logging.

        Args:
            cfg (Namespace): The configuration object containing the hyperparameters.

        Returns:
            Dict[str, Any]: A dictionary containing the parsed hyperparameters.

        """

        return {
            "embedding_dim": cfg.embedding_dim,
            "transformer": {
                "encoder_kwargs": vars(cfg.transformer.encoder_kwargs),
                "encoder_layer_kwargs": vars(cfg.transformer.encoder_layer_kwargs),
            },
        }

    def forward(
        self, batch: Batch, cond_ids_name: str, cat_ids_name: str, cond_embed_name: str
    ) -> StructuredForwardOutput:
        r"""
        Forward pass of the ConditionEmbedding module.

        Args:
            batch (Batch): The input batch containing the necessary data.
            cond_ids_name (str): The name of the condition IDs in the batch.
            cat_ids_name (str): The name of the category IDs in the batch.
            cond_embed_name (str): The name of the condition embedding in the batch.

        Returns:
            StructuredForwardOutput: The structured output of the forward pass.
        """

        cond_ids = batch[cond_ids_name]

        # Embed conditions.
        condition_embeddings = self._condition_embedding_module.embed(
            cond_ids=cond_ids, cat_ids=batch[cat_ids_name]
        )

        # Add repeated cls token to each sample in a batch.
        batch_size: int = condition_embeddings.shape[0]
        repeated_cls_pool_token: torch.Tensor = repeat(
            self._cls_pooling_token, "1 1 dim -> batch 1 dim", batch=batch_size
        )

        # Add cls token embedding representing condition set embedding.
        condition_embeddings = torch.cat(
            [
                repeated_cls_pool_token,
                condition_embeddings,
            ],
            dim=1,  # sequence dim
        )

        assert (
            torch.isnan(condition_embeddings).any().logical_not()
        ), "Some condition embeddings contain NaN values."

        # Specifying positions of the pad tokens.
        src_key_padding_mask = torch.cat(
            [torch.zeros((batch_size, 1)), (cond_ids == 0)], dim=1
        )

        # Apply transformer to condition embeddings.
        transformer_output = self._transformer_encoder(
            src=condition_embeddings, src_key_padding_mask=src_key_padding_mask
        )
        assert (
            torch.isnan(transformer_output).any().logical_not()
        ), "Some transformer outputs contain NaN values."

        # Extract the embedding of the cls token -> condition set embedding.
        batch[cond_embed_name] = transformer_output[:, 0, :]

        return format_structured_forward_output(batch=batch)
