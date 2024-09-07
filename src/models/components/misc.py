import torch
import torch.nn as nn
from functools import reduce, partial
from typing import TypeAlias, Dict, List, Callable, Set, Any
from utils.common_types import (
    Batch,
    ConfigStructure,
    StructuredForwardOutput,
    format_structured_forward_output,
)
from utils.config import validate_config_structure
from argparse import Namespace
from einops import rearrange, repeat

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


TensorReductor: TypeAlias = Callable[[torch.Tensor], torch.Tensor]

_TENSOR_REDUCTORS: Dict[str, TensorReductor] = {
    "mean": torch.mean,
    "logsumexp": torch.logsumexp,
}


def get_tensor_reductor(
    tensor_reductor_name: str, kwargs: Dict[str, Any]
) -> Callable[[torch.Tensor], torch.Tensor]:
    tensor_reductor = _TENSOR_REDUCTORS.get(tensor_reductor_name, None)
    if tensor_reductor is not None:
        return partial(tensor_reductor, **kwargs)
    raise ValueError(
        f"The provided tensor_reductor_name {tensor_reductor_name} is wrong. Must be one of [{', '.join(list(_TENSOR_REDUCTORS.keys()))}]."
    )


class AggregateDataAdapter(nn.Module):
    r"""
    A module for aggregating data in a batch using a specified aggregation type.
    Used for fusing data from previous modules.

    Args:
        aggregation_type (str): The type of aggregation to be performed.
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
        "kwargs": Namespace,
    }

    def __init__(self, cfg: Namespace) -> None:
        super(AggregateDataAdapter, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._aggregate: TensorAggregator = get_tensor_aggregator(
            aggregation_type=cfg.aggregation_type, kwargs=vars(cfg.kwargs)
        )

    def forward(
        self, batch: Batch, batch_aggr_def: Namespace
    ) -> StructuredForwardOutput:
        batch_aggr_def = vars(batch_aggr_def)

        to_be_removed: Set[str] = set()

        for aggregate_name, unnagregated_names in batch_aggr_def.items():
            to_be_aggregated: List[torch.Tensor] = []
            for data_name in unnagregated_names:
                to_be_removed.add(data_name)
                to_be_aggregated.append(batch[data_name])
            batch[aggregate_name] = self._aggregate(to_be_aggregated)

        for tbr in to_be_removed.difference(set(batch_aggr_def.keys())):
            del batch[tbr]

        return format_structured_forward_output(batch=batch, losses=[])


class TensorCloner(nn.Module):
    r"""
    A module for cloning tensors in a batch.

    Args:
        cfg (Namespace): The configuration namespace.

    Attributes:
        _config_structure (ConfigStructure): The configuration structure.

    Methods:
        forward(batch: Batch, data_name: str, clone_name: str) -> StructuredForwardOutput:
            Clones the tensor specified by `data_name` in the `batch` and assigns it to `clone_name`.
            Returns the formatted structured forward output.

    """

    _config_structure: ConfigStructure = {}

    def __init__(self, cfg: Namespace) -> None:
        super(TensorCloner, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

    def forward(
        self, batch: Batch, data_name: str, clone_name: str, clone: bool = True
    ) -> StructuredForwardOutput:
        if clone:
            batch[clone_name] = batch[data_name].clone()
        else:
            batch[clone_name] = batch[data_name]
        return format_structured_forward_output(batch=batch)


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

    _config_structure: ConfigStructure = {}

    def __init__(self, cfg: Namespace) -> None:
        super(BatchRearranger, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

    def forward(
        self, batch: Batch, data_name: str, pattern: str, **kwargs: Dict[str, Any]
    ) -> Batch:
        batch[data_name] = rearrange(tensor=batch[data_name], pattern=pattern, **kwargs)

        return format_structured_forward_output(batch=batch, losses=[])


class BatchRepeater(nn.Module):
    r"""
    A module for repeating tensors in a batch.

    Args:
        cfg (Namespace): The configuration namespace.

    Attributes:
        _config_structure (ConfigStructure): The configuration structure.

    Methods:
        forward(batch: Batch, data_name: str, pattern: str, **kwargs: Dict[str, Any]) -> StructuredForwardOutput:
            Repeats the tensor specified by `data_name` in the `batch` along the specified dimensions.
            Returns the formatted structured forward output.

    """

    _config_structure: ConfigStructure = {}

    def __init__(self, cfg: Namespace) -> None:
        super(BatchRepeater, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

    def forward(
        self, batch: Batch, data_name: str, pattern: str, **kwargs: Dict[str, Any]
    ) -> StructuredForwardOutput:
        batch[data_name] = repeat(tensor=batch[data_name], pattern=pattern, **kwargs)

        return format_structured_forward_output(batch=batch)


# plagiarised as for now from  https://github.com/janfreyberg/pytorch-revgrad/blob/master/src/pytorch_revgrad/functional.py
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
