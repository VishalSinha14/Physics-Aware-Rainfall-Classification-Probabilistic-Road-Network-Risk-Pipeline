import os
import xarray as xr

file_path = "data/raw/era5_2022_06.nc"

print("----- ERA5 FILE DIAGNOSTIC -----\n")

# 1️⃣ Check if file exists
if not os.path.exists(file_path):
    print("❌ File does NOT exist:", file_path)
    exit()

print("✅ File exists")

# 2️⃣ File size
size_mb = os.path.getsize(file_path) / (1024 * 1024)
print(f"📦 File size: {size_mb:.2f} MB")

# 3️⃣ Check first 4 bytes (magic number)
with open(file_path, "rb") as f:
    header = f.read(4)

print("🔎 First 4 bytes:", header)

if header.startswith(b'CDF'):
    print("👉 This is Classic NetCDF format")
elif header.startswith(b'\x89HDF'):
    print("👉 This is NetCDF4 (HDF5-based)")
elif header.startswith(b'GRIB'):
    print("👉 This is GRIB format (NOT NetCDF)")
else:
    print("👉 Unknown format")

print("\n----- TRYING TO OPEN WITH DIFFERENT ENGINES -----\n")

# 4️⃣ Try opening with netcdf4
try:
    ds = xr.open_dataset(file_path, engine="netcdf4")
    print("✅ Opened with netcdf4 engine")
    print("Dimensions:", ds.dims)
except Exception as e:
    print("❌ netcdf4 engine failed:", e)

# 5️⃣ Try opening with scipy
try:
    ds = xr.open_dataset(file_path, engine="scipy")
    print("✅ Opened with scipy engine")
    print("Dimensions:", ds.dims)
except Exception as e:
    print("❌ scipy engine failed:", e)

# 6️⃣ Try opening with cfgrib
try:
    ds = xr.open_dataset(file_path, engine="cfgrib")
    print("✅ Opened with cfgrib engine")
    print("Dimensions:", ds.dims)
except Exception as e:
    print("❌ cfgrib engine failed:", e)

print("\n----- DIAGNOSTIC COMPLETE -----")