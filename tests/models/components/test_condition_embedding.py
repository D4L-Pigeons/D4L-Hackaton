import torch
from src.models.components.condition_embedding import (
    DiscreteValuedConditionEmbedding,
    ConditionEmbeddingTransformer,
)
from argparse import Namespace
import torch
from argparse import Namespace
from src.models.components.condition_embedding import ConditionEmbeddingTransformer
import torch
from argparse import Namespace
from src.models.components.condition_embedding import ConditionEmbeddingTransformer


def test_discrete_valued_condition_embedding():
    cfg = Namespace(n_conditions=3, n_categories_per_condition=[2, 3, 4])
    embedding_dim = 5
    embedding = DiscreteValuedConditionEmbedding(cfg, embedding_dim)

    cond_ids = torch.tensor([[0, 1, 2]])
    cat_ids = torch.tensor([[1, 2, 3]])
    output = embedding.embed(cond_ids, cat_ids)

    assert output.shape == (1, 3, embedding_dim)


def test_discrete_valued_condition_embedding_with_zero_categories():
    cfg = Namespace(n_conditions=2, n_categories_per_condition=[1, 2])
    embedding_dim = 5
    embedding = DiscreteValuedConditionEmbedding(cfg, embedding_dim)

    cond_ids = torch.tensor([[0, 1]])
    cat_ids = torch.tensor([[0, 0]])
    output = embedding.embed(cond_ids, cat_ids)

    assert output.shape == (1, 2, embedding_dim)


def test_discrete_valued_condition_embedding_with_large_categories():
    cfg = Namespace(n_conditions=4, n_categories_per_condition=[1000, 2000, 3000, 4000])
    embedding_dim = 5
    embedding = DiscreteValuedConditionEmbedding(cfg, embedding_dim)

    cond_ids = torch.tensor([[0, 1, 2, 3]])
    cat_ids = torch.tensor([[999, 1999, 2999, 3999]])
    output = embedding.embed(cond_ids, cat_ids)

    assert output.shape == (1, 4, embedding_dim)


def test_discrete_valued_condition_embedding_with_negative_categories():
    cfg = Namespace(n_conditions=1, n_categories_per_condition=[1])
    embedding_dim = 5
    embedding = DiscreteValuedConditionEmbedding(cfg, embedding_dim)

    cond_ids = torch.tensor([[0]])
    cat_ids = torch.tensor([[1]])
    output = embedding.embed(cond_ids, cat_ids)

    assert output.shape == (1, 1, embedding_dim)


def test_condition_embedding_transformer():
    cfg = Namespace(
        cond_ids_name="cond_ids",
        cat_ids_name="cat_ids",
        cond_embed_name="cond_embed",
        embedding_dim=16,
        transformer=Namespace(
            encoder_kwargs=Namespace(num_layers=3),
            encoder_layer_kwargs=Namespace(nhead=2, dim_feedforward=256),
        ),
        embedding_module=Namespace(
            n_conditions=3,
            n_categories_per_condition=[2, 3, 4],
        ),
    )
    transformer = ConditionEmbeddingTransformer(cfg)

    batch = {
        "cond_ids": torch.tensor([[0, 1, 2]]),
        "cat_ids": torch.tensor([[1, 2, 3]]),
    }
    output = transformer(batch)

    assert "cond_embed" in output["batch"]
    assert output["batch"]["cond_embed"].shape == (1, cfg.embedding_dim)


def test_condition_embedding_transformer_with_large_embedding_dim():
    cfg = Namespace(
        cond_ids_name="cond_ids",
        cat_ids_name="cat_ids",
        cond_embed_name="cond_embed",
        embedding_dim=10,
        transformer=Namespace(
            encoder_kwargs=Namespace(num_layers=3),
            encoder_layer_kwargs=Namespace(nhead=2, dim_feedforward=256),
        ),
        embedding_module=Namespace(
            n_conditions=2,
            n_categories_per_condition=[1, 2],
        ),
    )
    transformer = ConditionEmbeddingTransformer(cfg)

    batch = {
        "cond_ids": torch.tensor([[0, 1], [1, 1]]),
        "cat_ids": torch.tensor([[0, 0], [1, 1]]),
    }
    output = transformer(batch)

    assert "cond_embed" in output["batch"]
    assert output["batch"]["cond_embed"].shape == (2, cfg.embedding_dim)
    assert torch.isnan(output["batch"]["cond_embed"]).any().logical_not()
