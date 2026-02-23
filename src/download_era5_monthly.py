import cdsapi
import calendar

c = cdsapi.Client()

year = 2022
month = '06'

num_days = calendar.monthrange(year, int(month))[1]
days = [f"{d:02d}" for d in range(1, num_days + 1)]

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            'total_precipitation',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            '2m_temperature',
            'surface_pressure',
            'volumetric_soil_water_layer_1'
        ],
        'year': str(year),
        'month': month,
        'day': days,
        'time': [f"{h:02d}:00" for h in range(24)],
        'area': [26.5, 104.5, 18, 117],
        'format': 'netcdf'
    },
    f'data/raw/era5_{year}_{month}.nc'
)

print("Download complete.")