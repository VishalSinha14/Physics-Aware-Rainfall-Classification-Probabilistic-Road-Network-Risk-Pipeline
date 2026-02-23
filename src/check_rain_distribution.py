import pandas as pd

print("Loading June temporal dataset...")

df = pd.read_csv("data/processed/era5_2022_06_temporal.csv")

print("Total samples:", len(df))

# Define heavy rain threshold
thresholds = [5, 10, 15, 20]

for threshold in thresholds:
    df["heavy_rain"] = (df["tp_mm"] >= threshold).astype(int)
    counts = df["heavy_rain"].value_counts()

    print("\nThreshold:", threshold, "mm/hr")
    print(counts)
    print("Percentage:")
    print((counts / len(df)) * 100)

df["heavy_rain"] = (df["tp_mm"] >= threshold).astype(int)

counts = df["heavy_rain"].value_counts()

print("\nHeavy Rain Threshold:", threshold, "mm/hr")
print("Class distribution:")
print(counts)

print("\nPercentage distribution:")
print((counts / len(df)) * 100)