import pandas as pd

# =========================
# LOAD DATASETS
# =========================

air = pd.read_csv("Data/air_quality_daily.csv")
cities = pd.read_csv("Data/cities.csv")
weather = pd.read_csv("Data/weather_daily.csv")


# =========================
# AIR QUALITY
# =========================

print("\n================ AIR QUALITY ================")

print("Shape:", air.shape)

print("\nColumns:")
print(air.columns.tolist())

print("\nFirst 5 rows:")
print(air.head())


# =========================
# CITIES
# =========================

print("\n================ CITIES ================")

print("Shape:", cities.shape)

print("\nColumns:")
print(cities.columns.tolist())

print("\nFirst 5 rows:")
print(cities.head())


# =========================
# WEATHER
# =========================

print("\n================ WEATHER ================")

print("Shape:", weather.shape)

print("\nColumns:")
print(weather.columns.tolist())

print("\nFirst 5 rows:")
print(weather.head())


# =========================
# MISSING VALUES
# =========================

print("\n================ MISSING VALUES ================")

print("\nAir Quality missing values:")
print(air.isnull().sum())

print("\nCities missing values:")
print(cities.isnull().sum())

print("\nWeather missing values:")
print(weather.isnull().sum())


# =========================
# DUPLICATES
# =========================

print("\n================ DUPLICATES ================")

print("Air Quality duplicates:", air.duplicated().sum())
print("Cities duplicates:", cities.duplicated().sum())
print("Weather duplicates:", weather.duplicated().sum())


# =========================
# CITY INFORMATION
# =========================

print("\n================ CITY INFORMATION ================")

print("Number of cities in air quality data:",
      air["city"].nunique())

print("\nCities:")
print(sorted(air["city"].unique()))


# =========================
# DATE INFORMATION
# =========================

print("\n================ DATE INFORMATION ================")

print("Air Quality date range:")
print(air["date"].min(), "to", air["date"].max())

print("\nWeather date range:")
print(weather["date"].min(), "to", weather["date"].max())


# =========================
# AQI INFORMATION
# =========================

print("\n================ AQI INFORMATION ================")

print("AQI statistics:")
print(air["india_aqi"].describe())