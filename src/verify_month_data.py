import xarray as xr

print("---- VERIFYING EXTRACTED INSTANT FILE ----")

instant_path = "data/raw/era5_2022_06_extracted/data_stream-oper_stepType-instant.nc"
accum_path = "data/raw/era5_2022_06_extracted/data_stream-oper_stepType-accum.nc"

# Open instant file
ds_instant = xr.open_dataset(instant_path)

print("\nInstant file dimensions:")
print(ds_instant.dims)

print("\nInstant time range:")
print("Start:", str(ds_instant.valid_time.min().values))
print("End  :", str(ds_instant.valid_time.max().values))
print("Total timestamps:", ds_instant.dims["valid_time"])

# Open accum file
ds_accum = xr.open_dataset(accum_path)

print("\nAccum file dimensions:")
print(ds_accum.dims)

print("\nAccum time range:")
print("Start:", str(ds_accum.valid_time.min().values))
print("End  :", str(ds_accum.valid_time.max().values))
print("Total timestamps:", ds_accum.dims["valid_time"])

print("\n---- VERIFICATION COMPLETE ----")