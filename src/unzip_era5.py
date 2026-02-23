import zipfile
import os

file_path = "data/raw/era5_2022_06.nc"

print("Extracting ZIP file...")

with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall("data/raw/era5_2022_06_extracted")

print("Extraction complete.")