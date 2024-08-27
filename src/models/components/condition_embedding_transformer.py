from re import L
import torch
import torch.nn as nn
from argparse import Namespace
from src.utils.common_types import ConfigStructure
from src.utils.config import validate_config_structure
from typing import List

# class ContinousValuedConditionEmbedding(nn.Module):
#     def __init__(self, cfg: Namespace) -> None:
#         self.token_embedding


class DiscreteValuedConditionEmbedding(nn.Module):
    _config_structure: ConfigStructure = {
        "n_conditions": int,
        "n_categories_per_condition": [int],
    }

    def __init__(self, cfg: Namespace, embedding_dim: int) -> None:
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._condition_embeddings: nn.Embedding = nn.Embedding(
            num_embeddings=cfg.n_conditions + 1, padding_idx=0
        )
        nn.init.xavier_normal_(self._condition_embedding)

        self._cat_embed_start_per_cond: List[int] = []
        n_categories_total: int = self._setup_cat_embedding_variables(cfg=cfg)
        self._category_embeddigns: nn.Embedding = nn.Embedding(
            num_embeddings=n_categories_total,
            embedding_dim=embedding_dim,
        )
        nn.init.xavier_normal_(self._category_embeddigns.weight)

    def _setup_cat_embedding_variables(self, cfg: Namespace) -> int:
        assert (
            len(cfg.n_categories_per_condition) == cfg.n_conditions
        ), "The number of entries in n_categories_per_condition must be equal to n_conditions."

        cumsum: int = 0
        for n_categories in cfg.cat_embed_start_per_cond:
            self._cat_embed_start_per_cond.append(cumsum)
            cumsum += n_categories

        return cumsum

    def embed(self, cat_ids: torch.Tensor) -> torch.Tensor:
        cat_ids = torch.LongTensor(cat_ids)
        return self._condition_embedding + self._category_embeddigns(cat_ids)


# class ConditionEmbeddingTransformer
