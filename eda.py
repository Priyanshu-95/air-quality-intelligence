import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# LOAD MASTER DATASET
# ==========================================

df = pd.read_csv("Data/master_air_quality.csv")

print("Dataset shape:", df.shape)


# ==========================================
# BASIC AQI STATISTICS
# ==========================================

print("\n========== AQI STATISTICS ==========")

print(df["india_aqi"].describe())


# ==========================================
# AVERAGE AQI BY CITY
# ==========================================

city_aqi = (
    df.groupby("city")["india_aqi"]
    .mean()
    .sort_values(ascending=False)
)

print("\n========== TOP 10 CITIES BY AVERAGE AQI ==========")

print(city_aqi.head(10))


# ==========================================
# 1. AQI DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["india_aqi"],
    bins=40,
    kde=True
)

plt.title("Distribution of India AQI")
plt.xlabel("AQI")
plt.ylabel("Number of Days")

plt.tight_layout()
plt.show()


# ==========================================
# 2. TOP 10 CITIES BY AQI
# ==========================================

plt.figure(figsize=(10, 6))

city_aqi.head(10).sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Cities by Average AQI")
plt.xlabel("Average AQI")
plt.ylabel("City")

plt.tight_layout()
plt.show()


# ==========================================
# 3. PM2.5 VS AQI
# ==========================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df.sample(min(5000, len(df))),
    x="pm2_5_mean",
    y="india_aqi",
    alpha=0.5
)

plt.title("PM2.5 vs India AQI")
plt.xlabel("PM2.5")
plt.ylabel("India AQI")

plt.tight_layout()
plt.show()


# ==========================================
# 4. PM10 VS AQI
# ==========================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df.sample(min(5000, len(df))),
    x="pm10_mean",
    y="india_aqi",
    alpha=0.5
)

plt.title("PM10 vs India AQI")
plt.xlabel("PM10")
plt.ylabel("India AQI")

plt.tight_layout()
plt.show()


# ==========================================
# 5. HUMIDITY VS AQI
# ==========================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df.sample(min(5000, len(df))),
    x="relative_humidity_2m_mean",
    y="india_aqi",
    alpha=0.5
)

plt.title("Humidity vs India AQI")
plt.xlabel("Relative Humidity (%)")
plt.ylabel("India AQI")

plt.tight_layout()
plt.show()


# ==========================================
# 6. WIND SPEED VS AQI
# ==========================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df.sample(min(5000, len(df))),
    x="wind_speed_10m_max",
    y="india_aqi",
    alpha=0.5
)

plt.title("Wind Speed vs India AQI")
plt.xlabel("Wind Speed")
plt.ylabel("India AQI")

plt.tight_layout()
plt.show()