import inspect
from functools import partial
from typing import Dict, Any
import neptune
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from argparse import Namespace

from utils.common_types import ConfigStructure
from utils.config import validate_config_structure
from utils.evaluation.plots import get_plotting_function_taking_dataloader


class NeptunePlotLogCallback(Callback):
    _config_structure: ConfigStructure = {
        "plotting_func": {"name": str, "kwargs": Namespace},
        "command_name": str,
        "neptune_plot_log_path": str,
        "plot_file_base_name": str,
        "command_dynamic_kwargs": Namespace,
    }

    def __init__(self, cfg: Namespace) -> None:
        super(NeptunePlotLogCallback, self).__init__()
        validate_config_structure(cfg=cfg, config_structure=self._config_structure)

        self._plotting_function = get_plotting_function_taking_dataloader(
            plotting_func_name=cfg.plotting_func.name, **vars(cfg.plotting_func.kwargs)
        )
        self._command_name = cfg.command_name
        self._plot_file_base_name = cfg.plot_file_base_name
        self._neptune_plot_log_path = cfg.neptune_plot_log_path
        self._command_dynamic_kwargs: Dict[str, Namespace] = vars(
            cfg.command_dynamic_kwargs
        )

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        val_dataloader = trainer.val_dataloaders
        run_processing_command_method = getattr(
            pl_module, "run_processing_command", None
        )

        assert (
            run_processing_command_method is not None
        ), "Provided pl_module has no attribute named run_processing_command."
        assert inspect.ismethod(run_processing_command_method) or inspect.isfunction(
            run_processing_command_method
        ), f"Attribute run_processing_command of pl_module is not a method."

        partial_run_processing_command_method = partial(
            run_processing_command_method,
            command_name=self._command_name,
            dynamic_kwargs=self._command_dynamic_kwargs,
            reset_loss_manager=True,
        )

        pl_module.train(False)

        figs = self._plotting_function(
            processing_function=partial_run_processing_command_method,
            dataloader=val_dataloader,
        )

        for fig_filename_end, fig in figs.items():
            # Log the plot to Neptune
            if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
                trainer.logger.experiment[
                    f"{self._neptune_plot_log_path}/{self._plot_file_base_name}-{fig_filename_end}-{trainer.current_epoch}"
                ].upload(neptune.types.File.as_image(fig))

            # Close the figure to release memory
            plt.close(fig)


_CALLBACKS: Dict[str, pl.Callback] = {"neptune_plot": NeptunePlotLogCallback}


def get_callback(
    callback_name: str, cfg: Namespace, kwargs: Dict[str, Any]
) -> pl.Callback:
    callback = _CALLBACKS.get(callback_name, None)

    if callback is not None:
        return callback(cfg=cfg, **kwargs)

    raise ValueError(
        f"The provided callback_name {callback_name} is invalid. Must be one of {list(_CALLBACKS.keys())}"
    )
