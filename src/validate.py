from src.metrics.metrics import *


def validate(config):
    # cross_validation_metrics.to_json(config.results_path / "metrics.json", indent=4)
    print(f"Metrics saved to: {config.results_path / 'metrics.json'}")


if __name__ == "__main__":
    validate()
