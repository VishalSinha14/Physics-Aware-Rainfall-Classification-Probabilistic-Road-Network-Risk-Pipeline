import xarray as xr
import numpy as np
import pandas as pd
import sys

def process_month(year, month):

    instant_path = f"data/raw/era5_{year}_{month:02d}_extracted/data_stream-oper_stepType-instant.nc"
    accum_path   = f"data/raw/era5_{year}_{month:02d}_extracted/data_stream-oper_stepType-accum.nc"

    ds_instant = xr.open_dataset(instant_path)
    ds_accum   = xr.open_dataset(accum_path)

    ds = xr.merge([ds_instant, ds_accum])

    print(f"{year}-{month:02d} merged dims:", ds.dims)

    ds["tp_mm"] = ds["tp"] * 1000
    ds["t2m_c"] = ds["t2m"] - 273.15
    ds["wind_speed"] = np.sqrt(ds["u10"]**2 + ds["v10"]**2)

    df = ds[["tp_mm","t2m_c","wind_speed","sp","swvl1"]].to_dataframe().reset_index()
    df = df.rename(columns={"valid_time":"time"})

    output_path = f"data/processed/era5_{year}_{month:02d}_raw.csv"
    df.to_csv(output_path, index=False)

    print(f"{year}-{month:02d} saved:", df.shape)


if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    process_month(year, month)

#python src/process_month_generic.py 2022 8