from pathlib import Path

# Data paths
DATA_PATH: Path = Path(__file__).parent.parent.parent / "data"
ANNDATA_FILENAME: str = "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad"
ANNDATA_PATH: Path = DATA_PATH / ANNDATA_FILENAME

# Config paths
CONFIG_PATH: Path = Path(__file__).parent.parent / "config"

# Results paths
RESULTS_PATH: Path = Path(__file__).parent.parent / "results"

# Logs paths
LOGS_PATH: Path = Path(__file__).parent.parent / "logs"
