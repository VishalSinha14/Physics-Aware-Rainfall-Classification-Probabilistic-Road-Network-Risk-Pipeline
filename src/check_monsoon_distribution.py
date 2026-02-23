import pandas as pd

print("Loading monsoon temporal dataset...")

df = pd.read_csv("data/processed/era5_2022_monsoon_temporal.csv")

print("Total samples:", len(df))

thresholds = [5, 10, 15, 20]

for threshold in thresholds:
    df["heavy_rain"] = (df["tp_mm"] >= threshold).astype(int)
    counts = df["heavy_rain"].value_counts()

    print("\nThreshold:", threshold, "mm/hr")
    print(counts)
    print("Percentage:")
    print((counts / len(df)) * 100)