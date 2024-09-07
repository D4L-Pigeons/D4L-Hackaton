from argparse import Namespace
import pytorch_lightning as pl

from utils.common_types import ConfigStructure
from utils.pl_trainer.callbacks import get_callback
from utils.config import validate_config_structure

_config_structure: ConfigStructure = {
    "callbacks": [{"name": str, "cfg": Namespace, "kwargs": Namespace}],
    "num_nodes": (None, int),
    "devices": (None, int | str, [int]),
    "accelerator": (None, str),
    "kwargs": Namespace,
}


def get_trainer(
    cfg: Namespace,
    max_epochs: int | None,
    check_val_every_n_epoch: int | None,
    logger: pl.loggers,
) -> pl.Trainer:
    validate_config_structure(cfg=cfg, config_structure=_config_structure)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=max_epochs,
        check_val_every_n_epoch=check_val_every_n_epoch,
        callbacks=[
            get_callback(
                callback_name=callback_spec.name,
                cfg=callback_spec.cfg,
                kwargs=vars(callback_spec.kwargs),
            )
            for callback_spec in cfg.callbacks
        ],
        accelerator=cfg.accelerator if cfg.accelerator is not None else "auto",
        devices=cfg.devices if cfg.devices is not None else "auto",
        num_nodes=cfg.num_nodes if cfg.num_nodes is not None else 1,
        **vars(cfg.kwargs)
    )

    return trainer
