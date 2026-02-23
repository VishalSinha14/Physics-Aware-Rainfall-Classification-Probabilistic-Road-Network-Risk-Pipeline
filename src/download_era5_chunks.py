import cdsapi

c = cdsapi.Client()

year = "2022"
month = "06"

# 5-day chunks for June
day_chunks = [
    ["01","02","03","04","05"],
    ["06","07","08","09","10"],
    ["11","12","13","14","15"],
    ["16","17","18","19","20"],
    ["21","22","23","24","25"],
    ["26","27","28","29","30"]
]

for i, days in enumerate(day_chunks):

    print(f"\nDownloading chunk {i+1}... Days: {days}")

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
            "year": year,
            "month": month,
            "day": days,
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [26.5, 104.5, 18, 117],
            "data_format": "netcdf"
        },
        f"data/raw/era5_{year}_{month}_chunk{i+1}.zip"
    )

    print(f"Chunk {i+1} completed.")