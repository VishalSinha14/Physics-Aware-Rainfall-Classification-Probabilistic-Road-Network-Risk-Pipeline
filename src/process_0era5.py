import xarray as xr

# Load both files
ds_instant = xr.open_dataset("data/raw/era5_20220601_instant.nc")
ds_accum = xr.open_dataset("data/raw/era5_20220601_accum.nc")

# Merge datasets
ds = xr.merge([ds_instant, ds_accum])

print(ds)