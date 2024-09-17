r"""
This module contains custom PyTorch Lightning callbacks for logging plots to Neptune and managing learning rate monitoring.

Classes:
    NeptunePlotLogCallback:
        A callback for logging plots to Neptune during training.

Functions:
    get_callback(callback_name: str, cfg: Namespace, kwargs: Dict[str, Any]) -> pl.Callback:
        Retrieves the specified callback based on the provided name and configuration.
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
import neptune
import inspect
from functools import partial
from argparse import Namespace
from typing import Dict, Callable, Any

from utils.evaluation.plots import PltTreeTopNode, get_top_node
from utils.config import validate_config_structure
from utils.common_types import ConfigStructure


class NeptunePlotLogCallback(Callback):
    r"""
    A PyTorch Lightning callback to log plots to Neptune during the validation phase.

    Attributes:
        _config_structure (ConfigStructure): The expected structure of the configuration.
        _command_name (str): The name of the command to run.
        _command_dynamic_kwargs (Dict[str, Namespace]): Dynamic keyword arguments for the command.
        _neptune_plot_log_path (str): The path in Neptune where plots will be logged.
        _plt_tree (PltTreeTopNode): The top node of the plot tree.

    Methods:
        __init__(cfg: Namespace) -> None:
            Initializes the callback with the given configuration.

        on_validation_end(trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            Called at the end of the validation phase to generate and log plots to Neptune.
    """

    _config_structure: ConfigStructure = {
        "neptune_plot_log_path": str,
        "command": {"name": str, "dynamic_kwargs": Namespace},
        "plt_tree_top_node": {"top_node_name": str, "top_node_cfg": Namespace},
    }

    def __init__(self, cfg: Namespace) -> None:
        super(NeptunePlotLogCallback, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._command_name = cfg.command.name
        self._command_dynamic_kwargs: Dict[str, Namespace] = vars(
            cfg.command.dynamic_kwargs
        )

        self._neptune_plot_log_path = cfg.neptune_plot_log_path

        self._plt_tree: PltTreeTopNode = get_top_node(
            top_node_name=cfg.plt_tree_top_node.top_node_name,
            cfg=cfg.plt_tree_top_node.top_node_cfg,
        )

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:

        val_dataloader = trainer.val_dataloaders
        run_command_method = getattr(pl_module, "run_command", None)

        assert (
            run_command_method is not None
        ), "Provided pl_module has no attribute named run_command."
        assert inspect.ismethod(run_command_method) or inspect.isfunction(
            run_command_method
        ), f"Attribute run_command of pl_module is not a method."

        partial_run_command_method = partial(
            run_command_method,
            command_name=self._command_name,
            dynamic_kwargs=self._command_dynamic_kwargs,
            reset_loss_manager=True,
        )

        pl_module.train(False)

        figs = self._plt_tree.get_plots(
            dataloader=val_dataloader,
            proc_fn=partial_run_command_method,
        )

        for fig_filename_end, fig in figs.items():
            # Log the plot to Neptune
            if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
                trainer.logger.experiment[
                    f"{self._neptune_plot_log_path}/{fig_filename_end}-{trainer.current_epoch}"
                ].upload(neptune.types.File.as_image(fig))


def no_cfg_wrapper(pl_callback: pl.Callback) -> Callable:
    r"""
    Wraps a PyTorch Lightning callback to ignore the configuration parameter.

    This function takes a PyTorch Lightning callback and returns a new callback
    that ignores the `cfg` parameter, allowing the original callback to be used
    without requiring a configuration object.

    Args:
        pl_callback (pl.Callback): The original PyTorch Lightning callback to be wrapped.

    Returns:
        Callable: A new callback function that ignores the `cfg` parameter.
    """

    def wrapped_pl_callback(cfg: Namespace, **kwargs: Dict[str, Any]) -> pl.Callback:
        return pl_callback(**kwargs)

    return wrapped_pl_callback


_CALLBACKS: Dict[str, pl.Callback] = {
    "neptune_plot": NeptunePlotLogCallback,
    "lr_monitor": no_cfg_wrapper(LearningRateMonitor),
}


def get_callback(
    callback_name: str, cfg: Namespace, kwargs: Dict[str, Any]
) -> pl.Callback:
    r"""
    Retrieve a callback instance based on the provided callback name.

    Args:
        callback_name (str): The name of the callback to retrieve.
        cfg (Namespace): Configuration namespace to pass to the callback.
        kwargs (Dict[str, Any]): Additional keyword arguments to pass to the callback.

    Returns:
        pl.Callback: An instance of the requested callback.

    Raises:
        ValueError: If the provided callback_name is not found in the _CALLBACKS dictionary.
    """
    callback = _CALLBACKS.get(callback_name, None)

    if callback is not None:
        return callback(cfg=cfg, **kwargs)

    raise ValueError(
        f"The provided callback_name {callback_name} is invalid. Must be one of {list(_CALLBACKS.keys())}"
    )
