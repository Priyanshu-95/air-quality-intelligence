import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "Data/master_air_quality.csv"
MODEL_PATH = "models/best_air_quality_model.pkl"

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

print("Dataset shape:", df.shape)


# ============================================================
# 2. FEATURES
# ============================================================

features = [
    "pm2_5_mean",
    "pm2_5_max",
    "pm10_mean",
    "pm10_max",
    "no2_mean",
    "so2_mean",
    "o3_8h_max",
    "co_8h_max",
    "dust_mean",
    "aod_mean",
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "precipitation_sum",
    "precipitation_hours",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "elevation_m"
]

target = "india_aqi"


# ============================================================
# 3. DATE-SAFE TRAIN / TEST SPLIT
# ============================================================

df = df.sort_values("date").reset_index(drop=True)

unique_dates = sorted(df["date"].unique())

split_date_index = int(len(unique_dates) * 0.80)

split_date = unique_dates[split_date_index]

train_data = df[df["date"] < split_date]
test_data = df[df["date"] >= split_date]

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

print("\n========== TRAIN / TEST ==========")

print("Training records:", len(train_data))
print("Testing records:", len(test_data))

print(
    "Training period:",
    train_data["date"].min(),
    "to",
    train_data["date"].max()
)

print(
    "Testing period:",
    test_data["date"].min(),
    "to",
    test_data["date"].max()
)

print("Split date:", split_date)


# ============================================================
# 4. LOAD BEST MODEL
# ============================================================

best_model = joblib.load(MODEL_PATH)

print("\nBest model loaded:")
print(type(best_model).__name__)


# ============================================================
# 5. PREDICTIONS
# ============================================================

y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("\n========== BEST MODEL PERFORMANCE ==========")

print("MAE  :", round(mae, 4))
print("RMSE :", round(rmse, 4))
print("R²   :", round(r2, 4))


# ============================================================
# 6. ACTUAL VS PREDICTED AQI
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.25,
    s=15
)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=2
)

plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")

plt.title("Actual vs Predicted AQI - Gradient Boosting")

plt.grid(alpha=0.2)

plt.tight_layout()

plt.savefig(
    "outputs/actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 7. PREDICTION ERROR DISTRIBUTION
# ============================================================

errors = y_test - y_pred

plt.figure(figsize=(10, 6))

plt.hist(
    errors,
    bins=50,
    edgecolor="black"
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=2
)

plt.xlabel("Prediction Error (Actual - Predicted)")
plt.ylabel("Frequency")

plt.title("AQI Prediction Error Distribution")

plt.grid(alpha=0.2)

plt.tight_layout()

plt.savefig(
    "outputs/prediction_error_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 8. MODEL COMPARISON
# ============================================================

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=15,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
}


results = []

print("\n========== MODEL COMPARISON ==========")

for name, model in models.items():

    print("Evaluating:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    model_mae = mean_absolute_error(
        y_test,
        predictions
    )

    model_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    model_r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "MAE": model_mae,
        "RMSE": model_rmse,
        "R2": model_r2
    })


results_df = pd.DataFrame(results)

print("\n")
print(
    results_df.sort_values(
        "RMSE"
    ).to_string(index=False)
)


# ============================================================
# 9. MODEL COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["RMSE"]
)

plt.ylabel("RMSE")

plt.xlabel("Machine Learning Model")

plt.title("Model Comparison - RMSE")

plt.xticks(
    rotation=15
)

plt.grid(
    axis="y",
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    "outputs/model_comparison_rmse.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

if hasattr(best_model, "feature_importances_"):

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": best_model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=True
    )

    plt.figure(figsize=(10, 8))

    plt.barh(
        importance["Feature"],
        importance["Importance"]
    )

    plt.xlabel("Importance")

    plt.ylabel("Feature")

    plt.title(
        "Feature Importance - Gradient Boosting"
    )

    plt.grid(
        axis="x",
        alpha=0.2
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 11. SAVE TEST PREDICTIONS
# ============================================================

evaluation_data = test_data[
    [
        "date",
        "city",
        "india_aqi",
        "india_aqi_category",
        "dominant_pollutant"
    ]
].copy()

evaluation_data["predicted_aqi"] = y_pred

evaluation_data["prediction_error"] = (
    evaluation_data["india_aqi"] -
    evaluation_data["predicted_aqi"]
)

evaluation_data.to_csv(
    "outputs/test_predictions.csv",
    index=False
)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n========================================")
print("MODEL EVALUATION COMPLETED")
print("========================================")

print("\nBest Model: Gradient Boosting")

print("MAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R²  :", round(r2, 4))

print("\nFiles created inside outputs/:")

print("1. actual_vs_predicted.png")
print("2. prediction_error_distribution.png")
print("3. model_comparison_rmse.png")
print("4. feature_importance.png")
print("5. test_predictions.csv")

print("\nEvaluation completed successfully!")