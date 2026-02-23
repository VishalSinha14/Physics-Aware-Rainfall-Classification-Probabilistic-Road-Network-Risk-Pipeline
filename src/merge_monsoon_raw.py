import pandas as pd

print("Loading monthly raw datasets...")

df_june  = pd.read_csv("data/processed/era5_2022_06_raw.csv")
df_july  = pd.read_csv("data/processed/era5_2022_07_raw.csv")
df_aug   = pd.read_csv("data/processed/era5_2022_08_raw.csv")

print("June:", df_june.shape)
print("July:", df_july.shape)
print("August:", df_aug.shape)

# Concatenate
df = pd.concat([df_june, df_july, df_aug], ignore_index=True)

print("After merge:", df.shape)

# Convert time properly
df["time"] = pd.to_datetime(df["time"])

# Sort globally (CRITICAL)
df = df.sort_values(by=["latitude", "longitude", "time"])

print("After sorting:", df.shape)

df.to_csv("data/processed/era5_2022_monsoon_raw.csv", index=False)

print("Merged monsoon dataset saved.")