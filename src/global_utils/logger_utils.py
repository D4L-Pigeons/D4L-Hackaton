from mrunner.helpers import client_helper
from pytorch_lightning.loggers import NeptuneLogger

from src.global_utils.paths import LOGS_PATH


def get_loggers(config):
    loggers = []
    if config.neptune_logger:
        loggers.append(NeptuneLogger(run=client_helper.experiment_))
    if config.tensorboard_logger:
        from pytorch_lightning.loggers import TensorBoardLogger

        loggers.append(TensorBoardLogger(LOGS_PATH, name=config.model_name))

    return loggers
