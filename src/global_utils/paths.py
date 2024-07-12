from pathlib import Path

# Filenames
RAW_ANNDATA_FILENAME: str = (
    "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad"
)
RAW_ANNDATA_HIERARCHY_FILENAME: str = "second_level_hierarchy.h5ad"  # change data type
PREPROCESSED_ANNDATA_FILENAME: str = "preprocessed.h5ad"

# Data paths
# Path: "path_to_repo/repo_name/data", for example: '/Users/asia/Desktop/D4L-Hackaton/data/'
DATA_PATH: Path = Path(__file__).parent.parent.parent / "data"
RAW_DATA_PATH: Path = DATA_PATH / "raw"
PREPROCESSED_DATA_PATH: Path = DATA_PATH / "preprocessed"

RAW_ANNDATA_PATH: Path = DATA_PATH / RAW_DATA_PATH / RAW_ANNDATA_FILENAME
RAW_ANNDATA_HIERARCHY_PATH: Path = (
    DATA_PATH / RAW_DATA_PATH / RAW_ANNDATA_HIERARCHY_FILENAME
)

# Preprocessed data paths
PREPROCESSED_ANNDATA_PATH: Path = DATA_PATH / "preprocessed.h5ad"

PREPROCESSED_DATA_PATH: Path = DATA_PATH / "preprocessed"
PREPROCESSED_ANNDATA_PATH: Path = PREPROCESSED_DATA_PATH / PREPROCESSED_ANNDATA_FILENAME

# Data for transformer models paths
TRANSFORMER_DATA_PATH: Path = DATA_PATH / "data_transformer.pt"

# Config paths
CONFIG_PATH: Path = Path(__file__).parent.parent / "config"

# Results paths
RESULTS_PATH: Path = Path(__file__).parent.parent / "results"

# Logs paths
LOGS_PATH: Path = Path(__file__).parent.parent / "logs"
