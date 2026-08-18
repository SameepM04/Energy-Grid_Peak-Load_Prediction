import numpy as np


def add_features(df):
    """
    Create legitimate historical and temporal features.

    IMPORTANT:
    Forecast columns and target-derived renewable metrics are NOT
    used as predictive features.
    """
    df = df.copy()

    ts = df["Timestamp_UTC"]

    # -----------------------------
    # Temporal features
    # -----------------------------
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["day_of_year"] = ts.dt.dayofyear
    df["month"] = ts.dt.month
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )
    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    # -----------------------------
    # Historical load features
    # -----------------------------
    # 15-minute intervals:
    # 96  = 1 day
    # 672 = 1 week
    for lag in (1, 2, 4, 96, 672):
        df[f"load_lag_{lag}"] = (
            df["Load [MW]"].shift(lag)
        )

    # Rolling statistics use ONLY previous observations.
    previous_load = df["Load [MW]"].shift(1)

    df["load_roll_mean_4"] = (
        previous_load.rolling(4).mean()
    )

    df["load_roll_std_4"] = (
        previous_load.rolling(4).std()
    )

    df["load_roll_mean_96"] = (
        previous_load.rolling(96).mean()
    )

    df["load_roll_std_96"] = (
        previous_load.rolling(96).std()
    )

    return df


def make_model_matrix(df):
    """
    Construct the ML feature matrix.

    STRICT LEAKAGE PROTECTION:

    The model target is:
        Load [MW]

    The following are excluded because they are either:
    1. the target itself,
    2. mathematically derived from the target,
    3. synthetic forecast columns,
    4. forecast-error columns, or
    5. target-derived analytical indicators.
    """

    forbidden = {
        # =========================================================
        # TARGET
        # =========================================================
        "Load [MW]",

        # =========================================================
        # TIMESTAMP
        # =========================================================
        "Timestamp_UTC",

        # =========================================================
        # TARGET-DERIVED RENEWABLE METRICS
        #
        # renewable_coverage_calc_pct =
        #       renewable_total / Load * 100
        #
        # renewable_gap_calc_MW =
        #       Load - renewable_total
        #
        # These MUST NOT enter a model predicting Load.
        # =========================================================
        "renewable_total_calc_MW",
        "renewable_coverage_calc_pct",
        "renewable_gap_calc_MW",

        # Existing derived versions, if present
        "renewable_total_MW",
        "renewable_coverage_pct",
        "renewable_gap_MW",

        # =========================================================
        # SYNTHETIC FORECAST VARIABLES
        # =========================================================
        "load_forecast_MW",
        "solar_forecast_MW",
        "wind_onshore_forecast_MW",
        "wind_offshore_forecast_MW",

        # =========================================================
        # FORECAST ERROR VARIABLES
        # =========================================================
        "load_forecast_error_MW",
        "solar_forecast_error_MW",
        "wind_onshore_forecast_error_MW",
        "wind_offshore_forecast_error_MW",

        # =========================================================
        # OTHER TARGET/RESULT-DERIVED VARIABLES
        # =========================================================
        "peak_flag",
        "risk_score",
        "potentially_critical",
        "condition_group",
        "condition",
        "demand_level",
        "renewable_level",
    }

    # Only numeric columns can enter the ML matrix.
    numeric = df.select_dtypes(
        include="number"
    ).copy()

    # Select legitimate columns only.
    features = [
        column
        for column in numeric.columns
        if column not in forbidden
    ]

    X = numeric[features].copy()
    y = df["Load [MW]"]

    # Remove rows made unavailable by lag/rolling features.
    valid = (
        X.notna().all(axis=1)
        & y.notna()
    )

    X = X.loc[valid]
    y = y.loc[valid]

    # ---------------------------------------------------------
    # XGBoost-safe feature names
    # ---------------------------------------------------------
    rename_map = {}

    for column in X.columns:
        safe_name = (
            str(column)
            .replace("[", "")
            .replace("]", "")
            .replace("<", "")
            .replace(">", "")
            .replace(" ", "_")
            .replace("-", "_")
        )

        rename_map[column] = safe_name

    X = X.rename(
        columns=rename_map
    )

    # Guarantee unique names.
    if len(set(X.columns)) != len(X.columns):
        X.columns = [
            f"feature_{i}"
            for i in range(X.shape[1])
        ]

    return X, y, list(X.columns)