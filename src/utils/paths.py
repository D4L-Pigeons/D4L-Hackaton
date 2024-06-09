from pathlib import Path

# Data paths
# Path: "path_to_repo/repo_name/data", for example: '/Users/asia/Desktop/D4L-Hackaton/data/'
DATA_PATH: Path = Path(__file__).parent.parent.parent / "data"
ANNDATA_FILENAME: str = "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad"
ANNDATA_PATH: Path = DATA_PATH / ANNDATA_FILENAME
HIERARCHY_PATH: Path = DATA_PATH / "second_hierarchy.h5ad"  # change data type

# Preprocessed data paths
PREPROCESSED_DATA_PATH: Path = DATA_PATH / "preprocessed"
PREPROCESSED_ANNDATA_PATH: Path = PREPROCESSED_DATA_PATH / "preprocessed.h5ad"

PEARSON_RESIDUALS_ANNDATA_FILENAME: str = (
    "GSE194122_openproblems_neurips2021_cite_BMMC_processed_pearson_residuals.h5ad"
)
PEARSON_RESIDUALS_ANNDATA_PATH: Path = (
    PREPROCESSED_DATA_PATH / PEARSON_RESIDUALS_ANNDATA_FILENAME
)

# Data for transformer models paths
TRANSFORMER_DATA_PATH: Path = DATA_PATH / "data_transformer.pt"

# Config paths
CONFIG_PATH: Path = Path(__file__).parent.parent / "config"

# Results paths
RESULTS_PATH: Path = Path(__file__).parent.parent / "results"

# Logs paths
LOGS_PATH: Path = Path(__file__).parent.parent / "logs"
