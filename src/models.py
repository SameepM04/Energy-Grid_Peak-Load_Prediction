from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


def chronological_split(X, y, test_size=0.20):
    """Split without shuffling so future observations stay in the test set."""
    split = int(len(X) * (1 - test_size))

    return (
        X.iloc[:split],
        X.iloc[split:],
        y.iloc[:split],
        y.iloc[split:],
    )


def get_models(random_state=42):
    return {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            max_features="sqrt",
        ),

        "XGBoost": XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def train_and_evaluate(
    models,
    X_train,
    X_test,
    y_train,
    y_test,
):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    rows = []
    predictions = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        rows.append({
            "Model": name,
            "MAE_MW": mean_absolute_error(
                y_test, prediction
            ),
            "RMSE_MW": np.sqrt(
                mean_squared_error(
                    y_test, prediction
                )
            ),
            "R2": r2_score(
                y_test, prediction
            ),
            "N": len(y_test),
        })

        predictions[name] = prediction
        trained_models[name] = model

    results = pd.DataFrame(rows).sort_values(
        "RMSE_MW"
    )

    return results, predictions, trained_models
