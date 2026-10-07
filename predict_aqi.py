import pandas as pd
import joblib


# ============================================================
# 1. LOAD DATA AND MODEL
# ============================================================

DATA_PATH = "Data/master_air_quality.csv"
MODEL_PATH = "models/best_air_quality_model.pkl"

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")
print("Model:", type(model).__name__)


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


# ============================================================
# 3. SELECT ONE REAL RECORD
# ============================================================

# Using one real record from the dataset
sample = df.iloc[0]

input_data = pd.DataFrame(
    [[sample[feature] for feature in features]],
    columns=features
)


# ============================================================
# 4. PREDICT AQI
# ============================================================

prediction = model.predict(input_data)

predicted_aqi = prediction[0]


# ============================================================
# 5. AQI CATEGORY
# ============================================================

def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Satisfactory"

    elif aqi <= 200:
        return "Moderate"

    elif aqi <= 300:
        return "Poor"

    elif aqi <= 400:
        return "Very Poor"

    else:
        return "Severe"


category = get_aqi_category(predicted_aqi)


# ============================================================
# 6. DISPLAY RESULT
# ============================================================

print("\n========================================")
print("       AIR QUALITY PREDICTION")
print("========================================")

print("\nCity:", sample["city"])
print("Date:", sample["date"])

print("\nPollution Information")
print("---------------------")

print("PM2.5 :", sample["pm2_5_mean"])
print("PM10  :", sample["pm10_mean"])
print("NO2   :", sample["no2_mean"])
print("SO2   :", sample["so2_mean"])
print("O3    :", sample["o3_8h_max"])
print("CO    :", sample["co_8h_max"])

print("\nWeather Information")
print("--------------------")

print("Temperature :", sample["temperature_2m_mean"])
print("Humidity    :", sample["relative_humidity_2m_mean"])
print("Wind Speed  :", sample["wind_speed_10m_max"])
print("Rainfall    :", sample["precipitation_sum"])

print("\n========================================")
print("Actual AQI    :", round(sample["india_aqi"], 2))
print("Predicted AQI :", round(predicted_aqi, 2))
print("AQI Category  :", category)
print("========================================")


# ============================================================
# 7. PREDICTION ERROR
# ============================================================

error = abs(sample["india_aqi"] - predicted_aqi)

print("\nPrediction Error:", round(error, 2))

print("\nPrediction test completed successfully!")