from pathlib import Path

# Data paths
DATA_PATH: Path = Path(__file__).parent.parent.parent / "data"
ANNDATA_FILENAME: str = "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad"
ANNDATA_PATH: Path = DATA_PATH / ANNDATA_FILENAME
HIERARCHY_PATH: Path = DATA_PATH / "second_hierarchy.h5ad" # change data type

# Preprocessed data paths
PREPROCESSED_DATA_PATH: Path = DATA_PATH / "preprocessed"
PEARSON_RESIDUALS_ANNDATA_FILENAME: str = (
    "GSE194122_openproblems_neurips2021_cite_BMMC_processed_pearson_residuals.h5ad"
)
PEARSON_RESIDUALS_ANNDATA_PATH: Path = (
    PREPROCESSED_DATA_PATH / PEARSON_RESIDUALS_ANNDATA_FILENAME
)

LOG1P_ANNDATA_PATH_FILENAME: str = (
    "GSE194122_openproblems_neurips2021_cite_BMMC_processed_log1p.h5ad"
)
LOG1P_ANNDATA_PATH: Path = PREPROCESSED_DATA_PATH / LOG1P_ANNDATA_PATH_FILENAME

STD_ANNDATA_PATH_FILENAME: str = (
    "GSE194122_openproblems_neurips2021_cite_BMMC_processed_std.h5ad"
)
STD_ANNDATA_PATH: Path = PREPROCESSED_DATA_PATH / STD_ANNDATA_PATH_FILENAME

# Config paths
CONFIG_PATH: Path = Path(__file__).parent.parent / "config"

# Results paths
RESULTS_PATH: Path = Path(__file__).parent.parent / "results"

# Logs paths
LOGS_PATH: Path = Path(__file__).parent.parent / "logs"
