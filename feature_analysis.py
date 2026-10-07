import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load master dataset
df = pd.read_csv("Data/master_air_quality.csv")

# Features we are considering
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

# Calculate correlation with AQI
correlations = (
    df[features + ["india_aqi"]]
    .corr()["india_aqi"]
    .drop("india_aqi")
    .sort_values(ascending=False)
)

print("\n========== CORRELATION WITH AQI ==========\n")

print(correlations)


# Plot correlations
plt.figure(figsize=(10, 8))

correlations.sort_values().plot(
    kind="barh"
)

plt.title("Feature Correlation with India AQI")
plt.xlabel("Correlation")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()