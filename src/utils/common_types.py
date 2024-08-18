import torch
from typing import List, Dict, TypeAlias, TypedDict, Type


Batch: TypeAlias = Dict[str, torch.Tensor]


class StructuredLoss(TypedDict):
    data: torch.Tensor
    coef: float
    name: str
    aggregated: bool


def format_structured_loss(
    loss: torch.Tensor, coef: float, name: str, aggregated: bool
) -> StructuredLoss:
    return StructuredLoss(data=loss, coef=coef, name=name, aggregated=aggregated)


class StructuredForwardOutput(TypedDict):
    batch: Batch
    losses: List[StructuredLoss]


def format_structured_forward_output(
    batch: Batch, losses: List[StructuredLoss]
) -> StructuredForwardOutput:
    return StructuredForwardOutput(batch=batch, losses=losses)


ConfigStructure: TypeAlias = Dict[str, Type | Dict]
