import pandas as pd
import numpy as np

print("Loading merged monsoon dataset...")

df = pd.read_csv("data/processed/era5_2022_monsoon_raw.csv")

print("Initial shape:", df.shape)

df["time"] = pd.to_datetime(df["time"])

# Already sorted but we ensure again
df = df.sort_values(by=["latitude", "longitude", "time"])

group_cols = ["latitude", "longitude"]

print("Adding lag features...")

df["rain_lag1"] = df.groupby(group_cols)["tp_mm"].shift(1)
df["rain_lag2"] = df.groupby(group_cols)["tp_mm"].shift(2)

print("Adding rolling features...")

df["rain_roll3"] = df.groupby(group_cols)["tp_mm"].transform(
    lambda x: x.rolling(3, min_periods=3).sum()
)

df["rain_roll6"] = df.groupby(group_cols)["tp_mm"].transform(
    lambda x: x.rolling(6, min_periods=6).sum()
)

print("Dropping NaNs...")

df = df.dropna()

print("Final shape:", df.shape)

df.to_csv("data/processed/era5_2022_monsoon_temporal.csv", index=False)

print("Monsoon temporal dataset saved.")