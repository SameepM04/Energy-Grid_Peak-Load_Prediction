from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = PROJECT_ROOT / "data" / "Raw" / "Dataset.csv"
CLEAN_DATA = PROJECT_ROOT / "data" / "Cleaned" / "CleanDataset.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"

for folder in (OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, MODEL_DIR):
    folder.mkdir(parents=True, exist_ok=True)

TARGET = "Load [MW]"

HIGH_DEMAND_QUANTILE = 0.90
LOW_RENEWABLE_QUANTILE = 0.25

TEST_SIZE = 0.20
RANDOM_STATE = 42
