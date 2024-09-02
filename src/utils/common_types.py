import torch
from typing import List, Dict, TypeAlias, TypedDict, Type, Optional

Batch: TypeAlias = Dict[str, torch.Tensor]


class StructuredLoss(TypedDict):
    data: torch.Tensor
    coef: float
    name: str
    reduced: bool


def format_structured_loss(
    loss: torch.Tensor, coef: float, name: str, reduced: bool
) -> StructuredLoss:
    if reduced:
        assert (
            loss.shape == ()
        ), f"If the loss '{name}' is reduced is should have shape () but got {loss.shape}."
    return StructuredLoss(data=loss, coef=coef, name=name, reduced=reduced)


class StructuredForwardOutput(TypedDict):
    batch: Batch
    losses: List[StructuredLoss]


def format_structured_forward_output(
    batch: Batch, losses: Optional[List[StructuredLoss]] = None
) -> StructuredForwardOutput:
    r"""
    Formats data to StrucutredForwardOutput object.

    Args:
        batch (Batch):
        losses (List[StructuredLoss])
    """
    return StructuredForwardOutput(
        batch=batch, losses=losses if losses is not None else []
    )


ConfigStructure: TypeAlias = Dict[str, Type | Dict]
