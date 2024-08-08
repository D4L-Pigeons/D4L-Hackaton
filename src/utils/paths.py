from pathlib import Path


# Data paths
DATA_PATH: Path = Path(__file__).parent.parent.parent / "data"
RAW_DATA_PATH: Path = DATA_PATH / "raw"
PREPROCESSED_DATA_PATH: Path = DATA_PATH / "preprocessed"
# ------

# Preprocessed data paths
PREPROCESSED_DATA_PATH: Path = DATA_PATH / "preprocessed"
# ------

# Config paths
CONFIG_PATH: Path = Path(__file__).parent.parent / "config"
# ------

# Results paths
RESULTS_PATH: Path = Path(__file__).parent.parent / "results"
# ------

# Logs paths
LOGS_PATH: Path = Path(__file__).parent.parent / "logs"
# ------

# Emebeddings paths
EMBEDDINGS_PATH: Path = DATA_PATH / "embeddings"
# ------

# Test paths
TESTS_PATH: Path = Path(__file__).parent.parent.parent / "tests"
TESTS_DATA_PATH: Path = TESTS_PATH / "data"
# ------
