import xarray as xr
import numpy as np
import pandas as pd

instant_path = "data/raw/era5_2022_06_extracted/data_stream-oper_stepType-instant.nc"
accum_path = "data/raw/era5_2022_06_extracted/data_stream-oper_stepType-accum.nc"

# Load
ds_instant = xr.open_dataset(instant_path)
ds_accum = xr.open_dataset(accum_path)

# Merge
ds = xr.merge([ds_instant, ds_accum])

print("Merged dimensions:", ds.dims)

# Feature engineering
ds["tp_mm"] = ds["tp"] * 1000
ds["t2m_c"] = ds["t2m"] - 273.15
ds["wind_speed"] = np.sqrt(ds["u10"]**2 + ds["v10"]**2)

# Convert to dataframe
df = ds[["tp_mm","t2m_c","wind_speed","sp","swvl1"]].to_dataframe().reset_index()
df = df.rename(columns={"valid_time":"time"})

print("Final dataframe shape:", df.shape)

df.to_csv("data/processed/era5_2022_06_raw.csv", index=False)

print("June processed and saved.")