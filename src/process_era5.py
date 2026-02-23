import xarray as xr
import pandas as pd
import numpy as np

# Load both files
ds_instant = xr.open_dataset("data/raw/era5_20220601_instant.nc")
ds_accum = xr.open_dataset("data/raw/era5_20220601_accum.nc")

# Merge
ds = xr.merge([ds_instant, ds_accum])

# ---- Feature Engineering ----

# Convert precipitation from meters to mm
ds['tp_mm'] = ds['tp'] * 1000

# Convert temperature from Kelvin to Celsius
ds['t2m_c'] = ds['t2m'] - 273.15

# Compute wind speed
ds['wind_speed'] = np.sqrt(ds['u10']**2 + ds['v10']**2)

# ---- Convert to DataFrame ----
df = ds[['tp_mm','t2m_c','wind_speed','sp','swvl1']].to_dataframe().reset_index()
df = df.drop(columns=['number','expver'])

# Rename valid_time → time
df = df.rename(columns={'valid_time':'time'})

print(df.head())
print("Shape:", df.shape)

# Save
df.to_csv("data/processed/era5_20220601_processed.csv", index=False)

print("Processing Complete!")