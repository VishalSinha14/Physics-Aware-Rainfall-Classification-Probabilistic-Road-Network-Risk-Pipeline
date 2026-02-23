import cdsapi

c = cdsapi.Client()

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
        'year': '2022',
        'month': '06',
        'day': '01',
        'time': [f"{i:02d}:00" for i in range(24)],
        'area': [26.5, 104.5, 18, 117],  # North, West, South, East
        'format': 'netcdf'
    },
    'data/raw/era5_test_20220601.nc'
)

print("Download Complete!")