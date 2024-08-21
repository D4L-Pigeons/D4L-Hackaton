import torch
import torch.nn as nn
from functools import reduce, partial
from typing import TypeAlias, Dict, List, Callable, Set, Any
from src.utils.common_types import (
    Batch,
    StructuredForwardOutput,
    ConfigStructure,
    format_structured_forward_output,
)
from src.utils.config import validate_config_structure
from argparse import Namespace
from einops import rearrange

BatchAggregationDefinition: TypeAlias = Dict[str, List[str]]

TensorAggregator: TypeAlias = Callable[[List[torch.Tensor]], torch.Tensor]

_TENSOR_AGGREGATORS: Dict[
    str, Callable[[torch.Tensor | List[torch.Tensor] | Any], torch.Tensor]
] = {
    "concat": lambda x, y, dim: torch.concat([x, y], dim=dim),
    "sum": lambda x, y: torch.add(x, y),
}


def get_tensor_aggregator(
    aggregation_type: str, kwargs: Dict[str, Any]
) -> TensorAggregator:
    tensor_aggregator = _TENSOR_AGGREGATORS.get(aggregation_type, None)
    if tensor_aggregator is not None:
        return lambda tensor_list: reduce(
            partial(tensor_aggregator, **kwargs), tensor_list
        )
    raise ValueError(
        f"The provided aggregation_type {aggregation_type} is wrong. Must be one of {' ,'.join(list(_TENSOR_AGGREGATORS.keys()))}"
    )


class AggregateDataAdapter(nn.Module):
    r"""
    A module for aggregating data in a batch using a specified aggregation type.
    Used for fusing data from previous modules.

    Args:
        aggregation_type (str): The type of aggregation to be performed.
        batch_aggr_def (BatchAggregationDefinition): The definition of batch aggregation.
        kwargs (Dict[str, Any]): Additional keyword arguments.

    Attributes:
        _batch_aggr_def (BatchAggregationDefinition): The definition of batch aggregation.
        _aggregate (TensorAggregator): The tensor aggregator used for aggregation.

    Methods:
        forward(batch: Batch) -> StructuredForwardOutput:
            Performs the forward pass of the module.

    """

    _config_structure: ConfigStructure = {
        "aggregation_type": str,
        "batch_aggr_def": Namespace,
        "kwargs": Namespace,
    }

    def __init__(self, cfg: Namespace) -> None:
        super(AggregateDataAdapter, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._batch_aggr_def: BatchAggregationDefinition = cfg.batch_aggr_def
        self._aggregate: TensorAggregator = get_tensor_aggregator(
            aggregation_type=cfg.aggregation_type, kwargs=vars(cfg.kwargs)
        )

    def forward(self, batch: Batch) -> StructuredForwardOutput:
        to_be_removed: Set[str] = set()
        batch_aggr_def = vars(self._batch_aggr_def)
        for aggregate_name, unnagregated_names in batch_aggr_def.items():
            to_be_aggregated: List[torch.Tensor] = []
            for data_name in unnagregated_names:
                to_be_removed.add(data_name)
                to_be_aggregated.append(batch[data_name])
            batch[aggregate_name] = self._aggregate(to_be_aggregated)
        for tbr in to_be_removed.difference(set(batch_aggr_def.keys())):
            del batch[tbr]
        return format_structured_forward_output(batch=batch, losses=[])


class BatchRearranger(nn.Module):
    r"""
    A module for rearranging batches of data according to specified patterns.

    Args:
        cfg (Namespace): The configuration for the module.

    Attributes:
        _rearrange_patterns (List[Namespace]): The list of rearrange patterns.

    Methods:
        forward(batch: Batch) -> Batch:
            Rearranges the batch of data according to the specified patterns.

    """

    _config_structure: ConfigStructure = {
        "rearrange_patterns": [{"data_name": str, "pattern": str, "kwargs": Namespace}]
    }

    def __init__(self, cfg: Namespace) -> None:
        super(BatchRearranger, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._rearrange_patterns: List[Namespace] = cfg.rearrange_patterns

    def forward(self, batch: Batch) -> Batch:
        for rearrange_pattern in self._rearrange_patterns:
            batch[rearrange_pattern.data_name] = rearrange(
                tensor=batch[rearrange_pattern.data_name],
                pattern=rearrange_pattern.pattern,
                **vars(rearrange_pattern.kwargs),
            )
        return format_structured_forward_output(batch=batch, losses=[])


# plagiarised as for now from  https://github.com/janfreyberg/pytorch-revgrad/blob/master/src/pytorch_revgrad/functional.py
# class RevGrad(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input_, alpha_):
#         ctx.save_for_backward(input_, alpha_)
#         output = input_
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):  # pragma: no cover
#         grad_input = None
#         _, alpha_ = ctx.saved_tensors
#         if ctx.needs_input_grad[0]:
#             grad_input = -grad_output * alpha_
#         return grad_input, None


# revgrad = RevGrad.apply
