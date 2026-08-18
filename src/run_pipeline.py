import json
from pathlib import Path

import joblib
import pandas as pd

from config import (
    CLEAN_DATA,
    TABLE_DIR,
    MODEL_DIR,
    TEST_SIZE,
    RANDOM_STATE,
)
from data import load_data, validate_data
from features import add_features, make_model_matrix
from renewable import (
    calculate_renewable_metrics,
    add_four_conditions,
    identify_reliability_windows,
)
from models import (
    chronological_split,
    get_models,
    train_and_evaluate,
)
from eda import run_eda


def main():
    print("=== CAPSTONE MODEL PIPELINE ===")

    # ------------------------------------------------------------
    # 1. LOAD + VALIDATE
    # ------------------------------------------------------------
    df = load_data(CLEAN_DATA)
    validation = validate_data(df)

    print("\n[1/6] Dataset validated")
    print(validation)

    (TABLE_DIR / "validation.json").write_text(
        json.dumps(
            validation,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # 2. FEATURES + EDA
    # ------------------------------------------------------------
    df = add_features(df)
    df = calculate_renewable_metrics(df)

    run_eda(df)

    print("[2/6] Features + EDA figures created")

    # ------------------------------------------------------------
    # 3. RELIABILITY WINDOWS
    # ------------------------------------------------------------
    high_demand = df["Load [MW]"].quantile(0.90)
    low_renewable = (
        df["renewable_coverage_calc_pct"]
        .quantile(0.25)
    )

    df = add_four_conditions(
        df,
        high_demand,
        low_renewable,
    )

    (
        df,
        windows,
        thresholds,
    ) = identify_reliability_windows(
        df,
        high_demand_quantile=0.90,
        low_renewable_quantile=0.25,
    )

    windows.to_csv(
        TABLE_DIR / "reliability_windows.csv",
        index=False,
    )

    pd.Series(thresholds).to_json(
        TABLE_DIR / "reliability_thresholds.json"
    )

    df["condition"].value_counts().to_csv(
        TABLE_DIR / "four_conditions.csv"
    )

    print(
        "[3/6] Reliability windows identified:",
        len(windows),
    )

    # ------------------------------------------------------------
    # 4. MODEL MATRIX + CHRONOLOGICAL SPLIT
    # ------------------------------------------------------------
    X, y, features = make_model_matrix(df)

    pd.DataFrame({
        "feature": features
    }).to_csv(
        TABLE_DIR / "model_features.csv",
        index=False,
    )

    X_train, X_test, y_train, y_test = (
        chronological_split(
            X,
            y,
            TEST_SIZE,
        )
    )

    print(
        f"[4/6] Chronological split: "
        f"{len(X_train)} train / "
        f"{len(X_test)} test"
    )

    # ------------------------------------------------------------
    # 5. TRAIN MODELS
    # ------------------------------------------------------------
    models = get_models(RANDOM_STATE)

    (
        results,
        predictions,
        trained_models,
    ) = train_and_evaluate(
        models,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("[5/6] Models trained")
    print("\n=== MODEL RESULTS ===")
    print(
        results.to_string(
            index=False
        )
    )

    results.to_csv(
        TABLE_DIR / "model_results.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # 6. SAVE PREDICTIONS + MODELS
    # ------------------------------------------------------------
    prediction_table = pd.DataFrame({
        "Timestamp_UTC": df.loc[
            y_test.index,
            "Timestamp_UTC",
        ].values,
        "actual_load_mw": y_test.values,
    })

    for name, prediction in predictions.items():
        safe_name = (
            name.lower()
            .replace(" ", "_")
        )
        prediction_table[
            f"pred_{safe_name}_mw"
        ] = prediction

    prediction_table.to_csv(
        TABLE_DIR / "test_predictions.csv",
        index=False,
    )

    for name, model in trained_models.items():
        safe_name = (
            name.lower()
            .replace(" ", "_")
        )
        joblib.dump(
            model,
            MODEL_DIR / f"{safe_name}.joblib",
        )

    # Feature importance for tree models
    xgb_model = trained_models.get("XGBoost")

    if xgb_model is not None:
        importance = pd.DataFrame({
            "feature": features,
            "importance": xgb_model.feature_importances_,
        }).sort_values(
            "importance",
            ascending=False,
        )

        importance.to_csv(
            TABLE_DIR / "xgboost_feature_importance.csv",
            index=False,
        )

    df.to_csv(
        TABLE_DIR / "processed_dataset.csv",
        index=False,
    )

    # Summary
    summary = {
        "dataset_rows": len(df),
        "model_rows": len(X),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "reliability_windows": len(windows),
        "high_demand_threshold_mw": float(
            high_demand
        ),
        "low_renewable_threshold_pct": float(
            low_renewable
        ),
        "excluded_synthetic_forecast_columns": [
            "load_forecast_MW",
            "solar_forecast_MW",
            "wind_onshore_forecast_MW",
            "wind_offshore_forecast_MW",
            "load_forecast_error_MW",
        ],
    }

    (TABLE_DIR / "project_summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "[6/6] Results, predictions and models saved"
    )


if __name__ == "__main__":
    main()