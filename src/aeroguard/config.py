"""Central paths and dataset constants."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
MLFLOW_DIR = ARTIFACTS_DIR / "mlruns"
RAG_DIR = ARTIFACTS_DIR / "rag"
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
DASHBOARD_DIR = PROJECT_ROOT / "src" / "aeroguard" / "api" / "static"

NASA_DATASET_PAGE = (
    "https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data"
)
NASA_ARCHIVE_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
ARCHIVE_NAME = "CMAPSSData.zip"

FD001_FILES = (
    "train_FD001.txt",
    "test_FD001.txt",
    "RUL_FD001.txt",
)

OPERATIONAL_COLUMNS = [f"operational_setting_{number}" for number in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{number}" for number in range(1, 22)]
CMAPSS_COLUMNS = ["engine_id", "cycle", *OPERATIONAL_COLUMNS, *SENSOR_COLUMNS]

RANDOM_STATE = 42
DEFAULT_RUL_CAP = 125


def ensure_directories() -> None:
    """Create every generated-data directory used by the project."""

    for directory in (
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        MLFLOW_DIR,
        RAG_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
