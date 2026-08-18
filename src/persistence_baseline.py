import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Make src importable when running from the project root.
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from config import CLEAN_DATA, TABLE_DIR
from data import load_data
from features import add_features, make_model_matrix
from renewable import calculate_renewable_metrics


def main():

    print("=== PERSISTENCE BASELINE ===")

    # =========================================================
    # 1. LOAD DATA
    # =========================================================

    df = load_data(CLEAN_DATA)

    print(
        f"Dataset loaded: {len(df)} observations"
    )

    # =========================================================
    # 2. CREATE THE SAME FEATURE MATRIX USED BY ML MODELS
    # =========================================================
    #
    # This is important.
    #
    # The ML pipeline removes the first 672 observations because
    # it requires:
    #
    #     load_lag_672
    #
    # for the weekly historical feature.
    #
    # We use the SAME valid observations here so that the
    # persistence baseline and ML models are evaluated on exactly
    # the same chronological test set.
    # =========================================================

    df_features = add_features(df)

    df_features = calculate_renewable_metrics(
        df_features
    )

    X, y, features = make_model_matrix(
        df_features
    )

    # X and y have exactly the rows available to the ML models.
    valid_indices = X.index

    evaluation_df = df_features.loc[
        valid_indices
    ].copy()

    # =========================================================
    # 3. PERSISTENCE PREDICTION
    # =========================================================
    #
    # Persistence forecasting:
    #
    #     predicted_load(t) = actual_load(t-1)
    #
    # Since the dataset has 15-minute intervals,
    # this means:
    #
    #     current prediction = load 15 minutes earlier
    #
    # =========================================================

    evaluation_df[
        "persistence_prediction_MW"
    ] = (
        evaluation_df["Load [MW]"].shift(1)
    )

    # The first valid model row cannot have a persistence
    # prediction if its previous row isn't inside the same
    # evaluation dataframe.
    #
    # Therefore remove only that one unavailable prediction.
    evaluation_df = evaluation_df.dropna(
        subset=[
            "Load [MW]",
            "persistence_prediction_MW",
        ]
    ).copy()

    # =========================================================
    # 4. EXACT SAME 80/20 CHRONOLOGICAL SPLIT
    # =========================================================

    split_index = int(
        len(evaluation_df) * 0.80
    )

    test = evaluation_df.iloc[
        split_index:
    ].copy()

    y_true = test[
        "Load [MW]"
    ]

    y_pred = test[
        "persistence_prediction_MW"
    ]

    # =========================================================
    # 5. CALCULATE METRICS
    # =========================================================

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    # =========================================================
    # 6. DISPLAY RESULTS
    # =========================================================

    print()
    print(
        "=== PERSISTENCE BASELINE RESULTS ==="
    )

    print(
        f"MAE  : {mae:.6f} MW"
    )

    print(
        f"RMSE : {rmse:.6f} MW"
    )

    print(
        f"R²   : {r2:.6f}"
    )

    print(
        f"N    : {len(test)}"
    )

    # =========================================================
    # 7. SAVE RESULTS
    # =========================================================

    results = pd.DataFrame([
        {
            "Model": "Persistence Baseline",
            "MAE_MW": mae,
            "RMSE_MW": rmse,
            "R2": r2,
            "N": len(test),
        }
    ])

    output_file = (
        TABLE_DIR
        / "persistence_baseline_results.csv"
    )

    results.to_csv(
        output_file,
        index=False,
    )

    # =========================================================
    # 8. SAVE PREDICTIONS
    # =========================================================

    predictions = test[
        [
            "Timestamp_UTC",
            "Load [MW]",
            "persistence_prediction_MW",
        ]
    ].copy()

    predictions = predictions.rename(
        columns={
            "Load [MW]":
                "actual_load_mw",
        }
    )

    predictions.to_csv(
        TABLE_DIR
        / "persistence_predictions.csv",
        index=False,
    )

    # =========================================================
    # 9. FINAL MESSAGE
    # =========================================================

    print()
    print(
        "[DONE] Results saved to:"
    )

    print(
        output_file
    )

    print()
    print(
        "Persistence baseline uses the same "
        "chronological test observations as "
        "the ML experiment."
    )


if __name__ == "__main__":
    main()