import zipfile
import os
import sys

def unzip_month(year, month):

    zip_path = f"data/raw/era5_{year}_{month:02d}.nc"
    extract_folder = f"data/raw/era5_{year}_{month:02d}_extracted"

    os.makedirs(extract_folder, exist_ok=True)

    print(f"Extracting {zip_path}...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    print("Extraction complete.")


if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    unzip_month(year, month)
    
#python src/unzip_month.py 2022 8