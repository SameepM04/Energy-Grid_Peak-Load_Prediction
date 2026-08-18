from pathlib import Path
import pandas as pd

from config import CLEAN_DATA, RAW_DATA


def load_data(path=None):
    """Load the cleaned dataset, falling back to the raw dataset."""
    path = Path(path) if path else CLEAN_DATA
    if not path.exists():
        path = RAW_DATA

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found. Expected:\n{CLEAN_DATA}\n{RAW_DATA}"
        )

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if "Timestamp_UTC" not in df.columns:
        raise KeyError("Timestamp_UTC column not found.")
    if "Load [MW]" not in df.columns:
        raise KeyError("Load [MW] column not found.")

    df["Timestamp_UTC"] = pd.to_datetime(
        df["Timestamp_UTC"], utc=True, errors="coerce"
    )

    if df["Timestamp_UTC"].isna().any():
        raise ValueError("Invalid timestamps found.")

    return df.sort_values("Timestamp_UTC").reset_index(drop=True)


def validate_data(df):
    """Return structural validation statistics."""
    diffs = df["Timestamp_UTC"].diff().dropna()

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_timestamps": int(
            df["Timestamp_UTC"].duplicated().sum()
        ),
        "start": str(df["Timestamp_UTC"].min()),
        "end": str(df["Timestamp_UTC"].max()),
        "non_15_minute_intervals": int(
            (diffs != pd.Timedelta(minutes=15)).sum()
        ),
    }
