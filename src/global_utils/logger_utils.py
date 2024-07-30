from mrunner.helpers import client_helper
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

from src.global_utils.paths import LOGS_PATH


def get_loggers(config):
    return [
        TensorBoardLogger(LOGS_PATH, name=config.model_name),
        NeptuneLogger(run=client_helper.experiment_),
    ]
