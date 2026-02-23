import cdsapi
import calendar
import sys

def download_month(year, month):

    c = cdsapi.Client()

    num_days = calendar.monthrange(year, month)[1]
    days = [f"{d:02d}" for d in range(1, num_days + 1)]

    print(f"\nDownloading {year}-{month:02d}...")

    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "total_precipitation",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
                "surface_pressure",
                "volumetric_soil_water_layer_1"
            ],
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [26.5, 104.5, 18, 117],
            "data_format": "netcdf"
        },
        f"data/raw/era5_{year}_{month:02d}.nc"
    )

    print(f"{year}-{month:02d} download complete.")


if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    download_month(year, month)

#python src/download_single_month.py (year month) e.g. python src/download_single_month.py 2022 8