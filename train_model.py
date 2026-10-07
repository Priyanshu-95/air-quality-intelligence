import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("Data/master_air_quality.csv")

df["date"] = pd.to_datetime(df["date"])

print("Dataset shape:", df.shape)


# ============================================================
# 2. SELECT FEATURES
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
# 3. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

# Sort the complete dataset by date
df_sorted = df.sort_values("date").reset_index(drop=True)


# Get all unique dates
unique_dates = sorted(df_sorted["date"].unique())


# Use 80% of the dates for training
split_date_index = int(len(unique_dates) * 0.80)

split_date = unique_dates[split_date_index]


# Training data = everything BEFORE split date
train_data = df_sorted[
    df_sorted["date"] < split_date
]


# Testing data = split date and everything AFTER
test_data = df_sorted[
    df_sorted["date"] >= split_date
]


# ============================================================
# 4. CREATE X AND y
# ============================================================

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]


print("\n========== TRAIN / TEST SPLIT ==========")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining period:")
print(
    train_data["date"].min(),
    "to",
    train_data["date"].max()
)

print("\nTesting period:")
print(
    test_data["date"].min(),
    "to",
    test_data["date"].max()
)

print("\nSplit date:")
print(split_date)


# ============================================================
# 5. DEFINE MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            max_depth=15,
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
}


# ============================================================
# 6. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

trained_models = {}

print("\n========== MODEL TRAINING ==========")


for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train model
    model.fit(
        X_train,
        y_train
    )

    # Make predictions
    predictions = model.predict(
        X_test
    )

    # Calculate MAE
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    # Calculate RMSE
    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    # Calculate R²
    r2 = r2_score(
        y_test,
        predictions
    )

    # Store results
    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    # Store trained model
    trained_models[name] = model

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")


# ============================================================
# 7. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


# Sort models by RMSE
# Lower RMSE = better model

results_df = results_df.sort_values(
    by="RMSE"
)


print("\n========== MODEL COMPARISON ==========")

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 8. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]


print("\n========== BEST MODEL ==========")

print(
    "Best model:",
    best_model_name
)


# ============================================================
# 9. SAVE BEST MODEL
# ============================================================

model_data = {
    "model": best_model,
    "features": features
}


joblib.dump(
    models[best_model_name],
    "models/best_air_quality_model.pkl"
)


print("\nBest model saved successfully!")

print(
    "File:"
)

print(
    "models/best_air_quality_model.pkl"
)


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

if hasattr(
    best_model,
    "feature_importances_"
):

    importance = pd.DataFrame({

        "Feature": features,

        "Importance":
            best_model.feature_importances_

    })


    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )


    print(
        "\n========== FEATURE IMPORTANCE =========="
    )

    print(
        importance.to_string(
            index=False
        )
    )