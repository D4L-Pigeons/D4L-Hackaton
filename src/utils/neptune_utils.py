from typing import Any, Dict

import neptune.new as neptune
from mrunner.helpers import client_helper


def get_metrics_callback(config):
    if config.neptune_logger:
        metrics_callback = _log_with_neptune
    else:
        metrics_callback = _print_metrics


def _print_metrics(stage, step, metrics, throttle=None):
    for k, v in metrics.items():
        print("    %s:" % k, v)


class NeptuneLogger:
    DEFAULT_STAGE_NAME = "logs"

    def __init__(self) -> None:
        self.neptune_run: neptune.Run = client_helper.experiment_

    def log(self, stage: str, step: int, metric: str, value: Any):
        self._ensure_run()
        self.neptune_run[f"{stage}/{metric}"].log(value=value, step=step)
        print("    %s:" % metric, value)

    def log_dict(self, stage: str, step: int, metrics: Dict[str, Any]):
        for k, v in metrics.items():
            self.log(stage, step, k, v)

    def _ensure_run(self):
        if self.neptune_run is None:
            self.neptune_run: neptune.Run = client_helper.experiment_


NEPTUNE_LOGGER = NeptuneLogger()


def _log_with_neptune(stage: str, step: int, metrics: Dict[str, Any]):
    NEPTUNE_LOGGER.log_dict(stage, step, metrics)
