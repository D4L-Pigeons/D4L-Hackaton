r"""
This module provides various PyTorch components for manipulating and processing batches of data. 

Classes:
    AggregateDataAdapter: A module for aggregating data in a batch using a specified aggregation type.
    TensorCloner: A module for cloning tensors in a batch.
    BatchRearranger: A module for rearranging batches of data according to specified patterns.
    BatchRepeater: A module for repeating tensors in a batch.

Functions:
    _get_explicit_constraint: Returns a tensor aggregator function based on the specified aggregation type.
    _get_tensor_reductor: Retrieves a tensor reduction function based on the provided name and additional keyword arguments.


"""

import torch
import torch.nn as nn
from functools import reduce, partial
from einops import rearrange, repeat
from argparse import Namespace
from typing import TypeAlias, Dict, List, Callable, Set, Any

from utils.common_types import (
    Batch,
    ConfigStructure,
    StructuredForwardOutput,
    format_structured_forward_output,
)
from utils.config import validate_config_structure

BatchAggregationDefinition: TypeAlias = Dict[str, List[str]]

TensorAggregator: TypeAlias = Callable[[List[torch.Tensor]], torch.Tensor]

_TENSOR_AGGREGATORS: Dict[
    str, Callable[[torch.Tensor | List[torch.Tensor] | Any], torch.Tensor]
] = {
    "concat": lambda x, y, dim: torch.concat([x, y], dim=dim),
    "sum": lambda x, y: torch.add(x, y),
}


def _get_explicit_constraint(
    aggregation_type: str, kwargs: Dict[str, Any]
) -> TensorAggregator:
    r"""
    Returns a tensor aggregator function based on the specified aggregation type.

    Args:
        aggregation_type (str): The type of aggregation to be used. Must be one of the keys in _TENSOR_AGGREGATORS.
        kwargs (Dict[str, Any]): Additional keyword arguments to be passed to the tensor aggregator.

    Returns:
        TensorAggregator: A function that aggregates a list of tensors using the specified aggregation type.

    Raises:
        ValueError: If the provided aggregation_type is not found in _TENSOR_AGGREGATORS.
    """

    tensor_aggregator = _TENSOR_AGGREGATORS.get(aggregation_type, None)

    if tensor_aggregator is None:
        raise ValueError(
            f"The provided aggregation_type {aggregation_type} is wrong. Must be one of {' ,'.join(list(_TENSOR_AGGREGATORS.keys()))}"
        )

    return lambda tensor_list: reduce(partial(tensor_aggregator, **kwargs), tensor_list)


TensorReductor: TypeAlias = Callable[[torch.Tensor], torch.Tensor]

_TENSOR_REDUCTORS: Dict[str, TensorReductor] = {
    "mean": torch.mean,
    "logsumexp": torch.logsumexp,
}


def _get_tensor_reductor(
    tensor_reductor_name: str, kwargs: Dict[str, Any]
) -> Callable[[torch.Tensor], torch.Tensor]:
    r"""
    Retrieves a tensor reduction function based on the provided name and additional keyword arguments.

    Args:
        tensor_reductor_name (str): The name of the tensor reduction function to retrieve.
        kwargs (Dict[str, Any]): Additional keyword arguments to pass to the tensor reduction function.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: A partially applied tensor reduction function.

    Raises:
        ValueError: If the provided tensor_reductor_name is not found in the available tensor reduction functions.
    """

    tensor_reductor = _TENSOR_REDUCTORS.get(tensor_reductor_name, None)

    if tensor_reductor is None:
        raise ValueError(
            f"The provided tensor_reductor_name {tensor_reductor_name} is wrong. Must be one of [{', '.join(list(_TENSOR_REDUCTORS.keys()))}]."
        )

    return partial(tensor_reductor, **kwargs)


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

        self._aggregate: TensorAggregator = _get_explicit_constraint(
            aggregation_type=cfg.aggregation_type, kwargs=vars(cfg.kwargs)
        )

    def forward(
        self, batch: Batch, batch_aggr_def: Namespace
    ) -> StructuredForwardOutput:
        r"""
        Processes a batch of data by aggregating specified components and removing others.

        Args:
            batch (Batch): The input batch of data.
            batch_aggr_def (Namespace): A namespace defining which components of the batch
                                        should be aggregated and their corresponding names.

        Returns:
            StructuredForwardOutput: The processed batch and an empty list of losses.
        """

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
        r"""
        Processes a batch of data, optionally cloning a specified data entry.

        Args:
            batch (Batch): The input batch of data.
            data_name (str): The key in the batch to be processed.
            clone_name (str): The key under which the processed data will be stored.
            clone (bool, optional): If True, the data will be cloned. If False, the data will be referenced directly. Defaults to True.

        Returns:
            StructuredForwardOutput: The formatted output containing the processed batch.
        """

        # Clonning or "copying" depends on the clone argument.
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
        r"""
        Applies a rearrangement pattern to a specified tensor within a batch and returns the modified batch.

        Args:
            batch (Batch): The input batch containing the tensor to be rearranged.
            data_name (str): The key in the batch dictionary corresponding to the tensor to be rearranged.
            pattern (str): The rearrangement pattern to be applied to the tensor.
            **kwargs (Dict[str, Any]): Additional keyword arguments to be passed to the rearrangement function.

        Returns:
            Batch: The modified batch with the rearranged tensor.
        """

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
        r"""
        Processes a batch of data by repeating a tensor according to a specified pattern and returns a structured output.

        Args:
            batch (Batch): The input batch of data.
            data_name (str): The key in the batch dictionary corresponding to the tensor to be repeated.
            pattern (str): The pattern according to which the tensor should be repeated.
            **kwargs (Dict[str, Any]): Additional keyword arguments to be passed to the repeat function.

        Returns:
            StructuredForwardOutput: The structured output after processing the batch.
        """

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
