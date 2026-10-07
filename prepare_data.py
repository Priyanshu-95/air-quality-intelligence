import pandas as pd

# ==========================================
# 1. LOAD DATA
# ==========================================

air = pd.read_csv("Data/air_quality_daily.csv")
weather = pd.read_csv("Data/weather_daily.csv")
cities = pd.read_csv("Data/cities.csv")


# ==========================================
# 2. CONVERT DATE TO DATETIME
# ==========================================

air["date"] = pd.to_datetime(air["date"])
weather["date"] = pd.to_datetime(weather["date"])


# ==========================================
# 3. MERGE AIR QUALITY + WEATHER
# ==========================================

merged = pd.merge(
    air,
    weather,
    on=["date", "city"],
    how="left"
)


# ==========================================
# 4. MERGE CITY LOCATION INFORMATION
# ==========================================

merged = pd.merge(
    merged,
    cities,
    on="city",
    how="left"
)


# ==========================================
# 5. DISPLAY INFORMATION
# ==========================================

print("\n================ MERGED DATA ================")

print("Shape:", merged.shape)

print("\nColumns:")
print(merged.columns.tolist())

print("\nFirst 5 rows:")
print(merged.head())


# ==========================================
# 6. CHECK MISSING VALUES AFTER MERGE
# ==========================================

print("\n================ MISSING VALUES ================")

missing = merged.isnull().sum()

print(missing[missing > 0])


# ==========================================
# 7. CHECK CITY COUNT
# ==========================================

print("\n================ CITY INFORMATION ================")

print("Number of cities:", merged["city"].nunique())


# ==========================================
# 8. SAVE MASTER DATASET
# ==========================================

merged.to_csv(
    "Data/master_air_quality.csv",
    index=False
)

print("\nMaster dataset saved successfully!")

print("File: Data/master_air_quality.csv")