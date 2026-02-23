import pandas as pd
import numpy as np

print("Loading June dataset...")

# Load dataset
df = pd.read_csv("data/processed/era5_2022_06_raw.csv")

print("Initial shape:", df.shape)

# Convert time to datetime (important)
df["time"] = pd.to_datetime(df["time"])

# Sort properly (CRITICAL for temporal correctness)
df = df.sort_values(by=["latitude", "longitude", "time"])

print("Sorting complete.")

# Define spatial grouping
group_cols = ["latitude", "longitude"]

print("Adding lag features...")

# Lag features
df["rain_lag1"] = df.groupby(group_cols)["tp_mm"].shift(1)
df["rain_lag2"] = df.groupby(group_cols)["tp_mm"].shift(2)

print("Adding rolling accumulation features...")

# Rolling sums (using transform keeps index aligned)
df["rain_roll3"] = df.groupby(group_cols)["tp_mm"].transform(
    lambda x: x.rolling(window=3, min_periods=3).sum()
)

df["rain_roll6"] = df.groupby(group_cols)["tp_mm"].transform(
    lambda x: x.rolling(window=6, min_periods=6).sum()
)

print("Dropping NaN rows created by lag/rolling...")

# Drop rows where lag/rolling not available
df = df.dropna()

print("Final shape after temporal features:", df.shape)

# Save
output_path = "data/processed/era5_2022_06_temporal.csv"
df.to_csv(output_path, index=False)

print("Temporal features added successfully.")
print("Saved to:", output_path)