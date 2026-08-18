import sys
from pathlib import Path

import pandas as pd
import pytest


# Make src/ importable when PyCharm launches pytest.
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))


from config import CLEAN_DATA
from data import load_data
from features import (
    add_features,
    make_model_matrix,
)
from renewable import (
    calculate_renewable_metrics,
    identify_reliability_windows,
)


@pytest.fixture
def dataset():
    return load_data(CLEAN_DATA)


# ============================================================
# DATASET TESTS
# ============================================================

def test_dataset_exists():
    assert CLEAN_DATA.exists()


def test_dataset_shape(dataset):
    assert len(dataset) == 105216
    assert len(dataset.columns) == 17


def test_required_columns(dataset):
    required = {
        "Timestamp_UTC",
        "Load [MW]",
        "Solar [MW]",
        "Wind onshore [MW]",
        "Wind offshore [MW]",
        "Hydro Run-of-River [MW]",
        "Hydro water reservoir [MW]",
    }

    missing = required - set(dataset.columns)

    assert not missing, (
        f"Missing required columns: {missing}"
    )


def test_no_missing_values(dataset):
    assert (
        dataset.isna().sum().sum()
        == 0
    )


def test_no_duplicate_rows(dataset):
    assert (
        dataset.duplicated().sum()
        == 0
    )


def test_no_duplicate_timestamps(dataset):
    assert (
        dataset["Timestamp_UTC"]
        .duplicated()
        .sum()
        == 0
    )


def test_timestamp_continuity(dataset):
    differences = (
        dataset["Timestamp_UTC"]
        .diff()
        .dropna()
    )

    invalid_intervals = (
        differences
        != pd.Timedelta(minutes=15)
    ).sum()

    assert invalid_intervals == 0


# ============================================================
# RENEWABLE ANALYSIS TESTS
# ============================================================

def test_renewable_calculations(dataset):

    df = add_features(dataset)

    df = calculate_renewable_metrics(df)

    assert (
        "renewable_total_calc_MW"
        in df.columns
    )

    assert (
        "renewable_coverage_calc_pct"
        in df.columns
    )

    assert (
        "renewable_gap_calc_MW"
        in df.columns
    )


# ============================================================
# MODEL MATRIX TESTS
# ============================================================

def test_model_matrix(dataset):

    df = add_features(dataset)

    df = calculate_renewable_metrics(df)

    X, y, features = (
        make_model_matrix(df)
    )

    assert len(X) > 0

    assert len(X) == len(y)

    assert len(features) > 0

    # Target itself must not be present.
    assert (
        "Load_MW"
        not in features
    )

    assert (
        "Load [MW]"
        not in features
    )


# ============================================================
# SYNTHETIC FORECAST PROTECTION
# ============================================================

def test_synthetic_forecasts_excluded(dataset):

    df = add_features(dataset)

    df = calculate_renewable_metrics(df)

    X, y, features = (
        make_model_matrix(df)
    )

    forbidden = {
        "load_forecast_MW",
        "solar_forecast_MW",
        "wind_onshore_forecast_MW",
        "wind_offshore_forecast_MW",

        "load_forecast_error_MW",
        "solar_forecast_error_MW",
        "wind_onshore_forecast_error_MW",
        "wind_offshore_forecast_error_MW",
    }

    leakage = (
        forbidden.intersection(
            features
        )
    )

    assert not leakage, (
        "Synthetic forecast columns "
        f"entered the model: {leakage}"
    )


# ============================================================
# TARGET-DERIVED FEATURE PROTECTION
# ============================================================

def test_target_derived_features_excluded(dataset):

    df = add_features(dataset)

    df = calculate_renewable_metrics(df)

    X, y, features = (
        make_model_matrix(df)
    )

    forbidden = {
        # Direct target
        "Load [MW]",

        # Renewable quantities derived using target
        "renewable_total_calc_MW",
        "renewable_coverage_calc_pct",
        "renewable_gap_calc_MW",

        # Existing versions
        "renewable_total_MW",
        "renewable_coverage_pct",
        "renewable_gap_MW",

        # Analytical results
        "peak_flag",
        "risk_score",
        "potentially_critical",
        "condition_group",
        "condition",
        "demand_level",
        "renewable_level",
    }

    leakage = (
        forbidden.intersection(
            features
        )
    )

    assert not leakage, (
        "Target-derived feature leakage "
        f"detected: {leakage}"
    )


# ============================================================
# RELIABILITY PIPELINE TEST
# ============================================================

def test_reliability_pipeline(dataset):

    df = add_features(dataset)

    df = calculate_renewable_metrics(df)

    (
        processed,
        windows,
        thresholds,
    ) = identify_reliability_windows(
        df
    )

    assert (
        "potentially_critical"
        in processed.columns
    )

    assert len(windows) >= 0

    assert (
        thresholds[
            "high_demand_threshold_mw"
        ] > 0
    )

    assert (
        thresholds[
            "low_renewable_threshold_pct"
        ] > 0
    )