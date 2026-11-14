
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


TARGET = "CO2 Emissions(g/km)"

FEATURES = [
    "Make",
    "Vehicle Class",
    "Engine Size(L)",
    "Cylinders",
    "Transmission",
    "Fuel Type",
    "Fuel Consumption City (L/100 km)",
    "Fuel Consumption Hwy (L/100 km)",
    "Fuel Consumption Comb (L/100 km)"
]

CATEGORICAL = [
    "Make",
    "Vehicle Class",
    "Transmission",
    "Fuel Type"
]

NUMERICAL = [
    "Engine Size(L)",
    "Cylinders",
    "Fuel Consumption City (L/100 km)",
    "Fuel Consumption Hwy (L/100 km)",
    "Fuel Consumption Comb (L/100 km)"
]


def train_models(df):

    required = FEATURES + [TARGET]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}. "
            "Please verify the Excel column names."
        )

    data = df[required].copy()

    for col in NUMERICAL + [TARGET]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=[TARGET])

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, NUMERICAL),
        ("cat", categorical_pipeline, CATEGORICAL)
    ])

    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    }

    results = []
    trained_models = {}

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": name,
            "R2 Score": r2,
            "RMSE": rmse,
            "MAE": mae
        })

        trained_models[name] = pipeline

        print(f"\n{name}")
        print("R2 Score:", round(r2, 4))
        print("RMSE:", round(rmse, 2))
        print("MAE:", round(mae, 2))

    results_df = pd.DataFrame(results)

    best_name = results_df.sort_values(
        "RMSE", ascending=True
    ).iloc[0]["Model"]

    best_model = trained_models[best_name]

    os.makedirs("model_artifacts", exist_ok=True)

    joblib.dump(
        best_model,
        "model_artifacts/best_model.pkl"
    )

    results_df.to_json(
        "model_artifacts/model_metrics.json",
        orient="records",
        indent=4
    )

    print("\nBest Model:", best_name)
    print("\nModel comparison:")
    print(results_df)

    return best_model, results_df
