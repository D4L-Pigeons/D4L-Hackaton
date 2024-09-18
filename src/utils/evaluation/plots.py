r"""
This module provides various utilities for creating and managing plots for evaluation purposes. It includes functions for generating specific types of plots, as well as classes for organizing and managing these plots in a hierarchical structure.

Modules and Functions:
- explained_variance_ratio_barplot: Creates a bar plot to visualize the explained variance ratio.
- _get_plt_fn: Retrieves a plotting function by its name and optionally applies partial arguments.

Classes:
- PltTreeNode: Abstract base class for creating plot tree nodes.
- PltFnNode: Represents a node in a plot tree structure, responsible for generating plots based on provided configuration and data.
- PltFnNodeList: Manages a list of PltFnNode instances.
- PerformPCANode: Performs Principal Component Analysis (PCA) on a given dataset and integrates with a plotting tree node to generate plots.
- PerformUMAPNode: Performs UMAP dimensionality reduction on a given dataset and integrates with a plotting tree node.
- PerformTSNENode: Performs t-SNE dimensionality reduction on a given dataset and generates plots using a specified plotting node.
- PltTreeTopNode: Abstract base class that represents the top node of a plotting tree structure.
- NProcessedBatchesPltTreeTopNode: Handles plotting for a specified number of processed batches.

Global Variables:
- _PLT_FNS: Dictionary mapping plot function names to their corresponding functions and configurations.
- _PLT_TREE_NODES: Dictionary mapping plot tree node names to their corresponding classes.
- _TOP_NODES: Dictionary mapping top node names to their corresponding classes.

Functions:
- _get_plt_tree_node: Retrieves a plotting tree node function based on the provided node name.
- get_top_node: Retrieves the top node from a predefined set of nodes based on the provided name.

"""

import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.manifold import TSNE
from functools import partial
import abc
from argparse import Namespace
from typing import TypeAlias, Callable, Dict, Any, List

from utils.evaluation.plots_base import (
    barplot,
    features_components_corrplot,
    ColoredScatterplot,
    original_vs_reconstructed_plot,
    images_with_conditions_plot,
    pairplot,
)
from utils.config import validate_config_structure
from utils.common_types import ConfigStructure

PltFn: TypeAlias = Callable[[Any], plt.Figure]
TensorDictFn: TypeAlias = Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]


def explained_variance_ratio_barplot(
    height: np.ndarray, **kwargs: Dict[str, Any]
) -> plt.Figure:
    r"""
        Creates a bar plot to visualize the explained variance ratio.

    Parameters:
        height (np.ndarray): An array containing the heights of the bars, representing the explained variance ratios.
        **kwargs (Dict[str, Any]): Additional keyword arguments to pass to the barplot function.

    Returns:
        plt.Figure: The matplotlib Figure object containing the bar plot.
    """

    return barplot(height=height, x_ticks=list(range(len(height))), **kwargs)


_PLT_FNS: Dict[str, PltFn] = {
    "explained_variance_ratio_barplot": Namespace(
        partial=True, fn=explained_variance_ratio_barplot
    ),
    "features_components_corrplot": Namespace(
        partial=True, fn=features_components_corrplot
    ),
    "colored_scatterplot": Namespace(partial=False, fn=ColoredScatterplot),
    "original_vs_reconstructed_plot": Namespace(
        partial=True, fn=original_vs_reconstructed_plot
    ),
    "images_with_conditions_plot": Namespace(
        partial=True, fn=images_with_conditions_plot
    ),
    "pairplot": Namespace(partial=True, fn=pairplot),
}


def _get_plt_fn(plt_fn_name: str, **kwargs: Dict[str, Any]) -> PltFn:
    r"""
    Retrieve a plotting function by its name and optionally apply partial arguments.

    Args:
        plt_fn_name (str): The name of the plotting function to retrieve.
        **kwargs (Dict[str, Any]): Additional keyword arguments to partially apply to the plotting function.

    Returns:
        PltFn: The plotting function corresponding to the provided name, optionally partially applied with the given arguments.

    Raises:
        ValueError: If the provided plt_fn_name does not exist in the _PLT_FNS dictionary.
    """

    plt_fn = _PLT_FNS.get(plt_fn_name, None)

    if plt_fn is None:
        raise ValueError(
            f"The provided plt_fn_name {plt_fn_name} is wrong. Must be one of {' ,'.join(list(_PLT_FNS.keys()))}"
        )

    if plt_fn.partial:
        return partial(plt_fn.fn, **kwargs)

    return plt_fn.fn


class PltTreeNode(abc.ABC):
    r"""
    Abstract base class for creating plot tree nodes.

    Attributes:
        cfg (Namespace): Configuration namespace containing necessary parameters.

    Methods:
        __init__(cfg: Namespace) -> None:
            Initializes the plot tree node with the given configuration.
            This method must be implemented by subclasses.

        get_plots(data: Dict[str, np.ndarray]) -> Dict[str, plt.Figure]:
            Generates and returns a dictionary of plots based on the provided data.
            This method must be implemented by subclasses.
    """

    @abc.abstractmethod
    def __init__(self, cfg: Namespace) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_plots(self, data: Dict[str, np.ndarray]) -> Dict[str, plt.Figure]:
        raise NotImplementedError


class PltFnNode(PltTreeNode):
    r"""
    PltFnNode is a class that represents a node in a plot tree structure, responsible for generating plots based on provided configuration and data.

    Attributes:
        _config_structure (ConfigStructure): A dictionary defining the expected structure of the configuration.
        _plt_fn (PltFn): A plotting function obtained based on the configuration.
        _data_args_names (List[str]): A list of argument names to be used for plotting.
        _filename_comp (str): A string component used for naming the output plot files.

    Methods:
        __init__(cfg: Namespace) -> None:
            Initializes the PltFnNode with the given configuration.

        get_plots(data: Dict[str, np.ndarray]) -> Dict[str, plt.Figure]:
            Generates and returns plots based on the provided data.
    """

    _config_structure: ConfigStructure = {
        "name": str,
        "kwargs": Namespace,
        "data_args_names": [str],
        "filename_comp": str,
    }

    def __init__(self, cfg: Namespace) -> None:
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._plt_fn: PltFn = _get_plt_fn(plt_fn_name=cfg.name, **vars(cfg.kwargs))
        self._data_args_names: List[str] = cfg.data_args_names
        self._filename_comp: str = cfg.filename_comp

    def get_plots(self, data: Dict[str, np.ndarray]) -> Dict[str, plt.Figure]:
        data_args = [data[data_arg_name] for data_arg_name in self._data_args_names]

        return {self._filename_comp: self._plt_fn(*data_args)}


class PltFnNodeList(PltTreeNode):
    r"""
    PltFnNodeList is a class that extends PltTreeNode and is responsible for managing a list of PltFnNode instances.

    Attributes:
        _config_structure (ConfigStructure): Defines the expected structure of the configuration.
        _plt_fn_node_list (List[PltFnNode]): A list of PltFnNode instances initialized from the provided configuration.

    Methods:
        __init__(cfg: Namespace) -> None:
            Initializes the PltFnNodeList instance by validating the provided configuration and creating a list of PltFnNode instances.

        get_plots(data: Dict[str, np.ndarray]) -> Dict[str, plt.Figure]:
            Generates and returns a dictionary of plot figures by aggregating the plots from each PltFnNode in the list.
    """

    _config_structure: ConfigStructure = {"plt_fn_node_list": [Namespace]}

    def __init__(self, cfg: Namespace) -> None:
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._plt_fn_node_list: List[PltFnNode] = [
            PltFnNode(cfg=plt_fn_node_cfg) for plt_fn_node_cfg in cfg.plt_fn_node_list
        ]

    def get_plots(self, data: Dict[str, np.ndarray]) -> Dict[str, plt.Figure]:
        fig_dict = {}
        for plt_fn_node in self._plt_fn_node_list:
            fig_dict.update(plt_fn_node.get_plots(data=data))

        return fig_dict


_PLT_TREE_NODES: Dict[str, PltTreeNode] = {
    "plt_fn_node_list": PltFnNodeList,
}


def _get_plt_tree_node(plt_node_name: str, cfg: Namespace) -> PltTreeNode:
    r"""
    Retrieve a plotting tree node function based on the provided node name.

    Args:
        plt_node_name (str): The name of the plotting tree node to retrieve.
        cfg (Namespace): Configuration object to be passed to the plotting tree node function.

    Returns:
        PltTreeNode: The plotting tree node function corresponding to the provided name.

    Raises:
        ValueError: If the provided plt_node_name does not exist in the _PLT_TREE_NODES dictionary.
    """

    plt_fn_node = _PLT_TREE_NODES.get(plt_node_name, None)

    if plt_fn_node is None:
        raise ValueError(
            f"The provided plt_node_name'{plt_node_name}' is wrong. Must be one of {' ,'.join(list(_PLT_TREE_NODES.keys()))}"
        )

    return plt_fn_node(cfg=cfg)


class PerformPCANode(PltTreeNode):
    r"""
    PerformPCANode is a class that performs Principal Component Analysis (PCA) on a given dataset
    and integrates with a plotting tree node to generate plots.

    Attributes:
        _config_structure (ConfigStructure): The structure of the configuration expected by the node.
        _pca (PCA): The PCA instance used to perform dimensionality reduction.
        _plt_tree_node (PltTreeNode): The plotting tree node used to generate plots.
        _data_name (str): The name of the data to be transformed.

    Methods:
        __init__(cfg: Namespace) -> None:
            Initializes the PerformPCANode with the given configuration.

        get_plots(data: Dict[str, np.ndarray]) -> Dict[str, plt.Figure]:
            Transforms the data using PCA, updates the data dictionary with PCA results,
            and generates plots using the plotting tree node.
    """

    _config_structure: ConfigStructure = {
        "kwargs": Namespace,
        "data_name": str,
        "node_name": str,
        "node_cfg": Namespace,
    }

    def __init__(self, cfg: Namespace) -> None:
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._pca = PCA(**vars(cfg.kwargs))
        self._plt_tree_node: PltTreeNode = _get_plt_tree_node(
            plt_node_name=cfg.node_name, cfg=cfg.node_cfg
        )
        self._data_name: str = cfg.data_name

    def get_plots(self, data: Dict[str, np.ndarray]) -> Dict[str, plt.Figure]:
        data[self._data_name] = self._pca.fit_transform(X=data[self._data_name])
        data[self._data_name + "_components"] = self._pca.components_
        data[self._data_name + "_explained_variance_ratio"] = (
            self._pca.explained_variance_ratio_
        )

        return self._plt_tree_node.get_plots(data=data)


_PLT_TREE_NODES["perform_pca_node"] = PerformPCANode


class PerformUMAPNode(PltTreeNode):
    r"""
    PerformUMAPNode is a class that performs UMAP dimensionality reduction on a given dataset and integrates with a plotting tree node.

    Attributes:
        _config_structure (ConfigStructure): The expected structure of the configuration.
        _umap (UMAP): The UMAP instance used for dimensionality reduction.
        _plt_tree_node (PltTreeNode): The plotting tree node for generating plots.
        _data_name (str): The name of the data to be transformed.

    Methods:
        __init__(cfg: Namespace) -> None:
            Initializes the PerformUMAPNode with the given configuration.

        get_plots(data: Dict[str, np.ndarray]) -> Dict[str, plt.Figure]:
            Transforms the data using UMAP and generates plots using the plotting tree node.
    """

    _config_structure: ConfigStructure = {
        "kwargs": Namespace,
        "data_name": str,
        "node_name": str,
        "node_cfg": Namespace,
    }

    def __init__(self, cfg: Namespace) -> None:
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._umap = UMAP(**vars(cfg.kwargs))
        self._plt_tree_node: PltTreeNode = _get_plt_tree_node(
            plt_node_name=cfg.node_name, cfg=cfg.node_cfg
        )
        self._data_name: str = cfg.data_name

    def get_plots(self, data: Dict[str, np.ndarray]) -> Dict[str, plt.Figure]:
        data[self._data_name] = self._umap.fit_transform(X=data[self._data_name])

        return self._plt_tree_node.get_plots(data=data)


_PLT_TREE_NODES["perform_umap_node"] = PerformUMAPNode


class PerformTSNENode(PltTreeNode):
    r"""
    PerformTSNENode is a class that performs t-SNE dimensionality reduction on a given dataset and generates plots using a specified plotting node.

    Attributes:
        _config_structure (ConfigStructure): The expected structure of the configuration.
        _tsne (TSNE): The t-SNE model initialized with the provided configuration.
        _plt_tree_node (PltTreeNode): The plotting node used to generate plots.
        _data_name (str): The name of the data key in the input dictionary.

    Methods:
        __init__(cfg: Namespace) -> None:
            Initializes the PerformTSNENode with the provided configuration.

        get_plots(data: Dict[str, np.ndarray]) -> Dict[str, plt.Figure]:
            Performs t-SNE on the specified data and generates plots using the plotting node.
    """

    _config_structure: ConfigStructure = {
        "kwargs": Namespace,
        "data_name": str,
        "node_name": str,
        "node_cfg": Namespace,
    }

    def __init__(self, cfg: Namespace) -> None:
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._tsne = TSNE(**vars(cfg.kwargs))
        self._plt_tree_node: PltTreeNode = _get_plt_tree_node(
            plt_node_name=cfg.node_name, cfg=cfg.node_cfg
        )
        self._data_name: str = cfg.data_name

    def get_plots(self, data: Dict[str, np.ndarray]) -> Dict[str, plt.Figure]:
        data[self._data_name] = self._tsne.fit_transform(X=data[self._data_name])

        return self._plt_tree_node.get_plots(data=data)


_PLT_TREE_NODES["perform_tsne_node"] = PerformTSNENode


class PltTreeTopNode(abc.ABC):
    r"""
    PltTreeTopNode is an abstract base class that represents the top node of a plotting tree structure.

    Attributes:
        _config_structure (ConfigStructure): A dictionary defining the expected structure of the configuration.

    Methods:
        __init__(cfg: List[Namespace]) -> None:
            Initializes the PltTreeTopNode with the given configuration.
            Args:
                cfg (List[Namespace]): A list of Namespace objects containing the configuration for the plotting tree.

        get_plots(dataloader: torch.utils.data.DataLoader, proc_fn: TensorDictFn) -> Dict[str, plt.Figure]:
            Abstract method that must be implemented by subclasses to generate plots.
            Args:
                dataloader (torch.utils.data.DataLoader): A DataLoader object to provide data for plotting.
                proc_fn (TensorDictFn): A function to process the data.
            Returns:
                Dict[str, plt.Figure]: A dictionary where keys are plot names and values are matplotlib Figure objects.
    """

    _config_structure: ConfigStructure = {
        "plt_tree_cfg": [{"node_name": str, "node_cfg": Namespace}]
    }

    def __init__(self, cfg: List[Namespace]) -> None:
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._plt_fns_tree = [
            _get_plt_tree_node(
                plt_node_name=plt_fn_node.node_name, cfg=plt_fn_node.node_cfg
            )
            for plt_fn_node in cfg.plt_tree_cfg
        ]

    @abc.abstractmethod
    def get_plots(
        self,
        dataloader: torch.utils.data.DataLoader,
        proc_fn: TensorDictFn,
    ) -> Dict[str, plt.Figure]:
        raise NotImplementedError


class NProcessedBatchesPltTreeTopNode(PltTreeTopNode):
    r"""
    A class to handle plotting for a specified number of processed batches.

    Attributes:
        _config_structure (ConfigStructure): Configuration structure for the class.
        _n_batches (int): Number of batches to process.

    Methods:
        __init__(cfg: Namespace) -> None:
            Initializes the NProcessedBatchesPltTreeTopNode with the given configuration.

        get_plots(
            Generates plots from the processed batches using the provided dataloader and processing function.
    """

    _config_structure: ConfigStructure = {
        "n_batches": int
    } | PltTreeTopNode._config_structure

    def __init__(self, cfg: Namespace) -> None:
        super(NProcessedBatchesPltTreeTopNode, self).__init__(cfg=cfg)
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)
        self._n_batches: int = cfg.n_batches

    def get_plots(
        self,
        dataloader: torch.utils.data.DataLoader,
        proc_fn: TensorDictFn,
    ) -> Dict[str, plt.Figure]:
        dataloader_iter = iter(dataloader)
        batch_aggregate = proc_fn(batch=next(dataloader_iter))

        for batch, batch_idx in zip(dataloader_iter, range(self._n_batches - 1)):
            batch = proc_fn(batch=batch)

            for key in batch_aggregate.keys():
                batch_aggregate[key] = torch.cat(
                    [batch_aggregate[key], batch[key]], dim=0
                )

        plots: Dict[str, plt.Figure] = {}

        for plt_fn_node in self._plt_fns_tree:
            fig_dict = plt_fn_node.get_plots(batch_aggregate)
            plots.update(fig_dict)

        return plots


_TOP_NODES: Dict[str, PltTreeTopNode] = {"nproc_batch": NProcessedBatchesPltTreeTopNode}


def get_top_node(top_node_name: str, cfg: Namespace) -> PltTreeTopNode:
    r"""
    Retrieves the top node from a predefined set of nodes based on the provided name.

    Args:
        top_node_name (str): The name of the top node to retrieve.
        cfg (Namespace): Configuration object to be passed to the top node.

    Returns:
        PltTreeTopNode: The top node corresponding to the provided name.

    Raises:
        ValueError: If the provided top_node_name does not exist in the predefined set of nodes.
    """

    top_node = _TOP_NODES.get(top_node_name, None)

    if top_node is None:
        raise ValueError(
            f"The provided top_node_name '{top_node_name}' is wrong. Must be one of {' ,'.join(list(_TOP_NODES.keys()))}"
        )

    return top_node(cfg=cfg)
