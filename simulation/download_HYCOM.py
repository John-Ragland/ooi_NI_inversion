import fsspec
import xarray as xr
import numpy as np
import os
from tqdm import tqdm

# calculated in notebook
lat_idx = 3149
lon_idx = 2875
# hycom LAT/LON cooridnate is 1.569 km from midpoint
midpoint = (45.9472222222222, -129.991388888)

ftp_url = 'ftp://ftp.hycom.org/datasets/GLBy0.08/expt_93.0/data/hindcasts/'

os.system('rm file_list.txt')

file_list = open('file_list.txt', 'a+')
with open('file_list.txt', 'w') as f:
    for year in range(2018,2024):
        url = f'{ftp_url}{year}/'
        files = os.popen(f'curl --list-only {url}').read()
        f.writelines(files)

### Get list of temperature / salinity files
with open('file_list.txt', 'r') as f:
    files2 = f.readlines()
    
files = []
for file in files2:
    files.append(file[:-1])

temp_files = []
for file in files:
    if file[-7:-3] == 'ts3z':
        temp_files.append(file)




thredds_base = 'https://tds.hycom.org/thredds/dodsC/datasets/GLBy0.08/expt_93.0/data/hindcasts/'
first_loop = True
for k, file in enumerate(tqdm(temp_files)):
    yr = file[15:19]
    url = thredds_base + yr + '/' + file
    ds = xr.open_dataset(url, decode_times=False)
    ds_slice = ds.isel({'lat':lat_idx, 'lon':lon_idx}).chunk({'time':1, 'depth':40})[['water_temp','salinity']]
        
    if first_loop:
        first_loop=False
        ds_slice.to_zarr('/datadrive/HYCOM_data/AxialSeamount_Caldera.zarr', mode='w-')
    else:
        ds_slice.to_zarr('/datadrive/HYCOM_data/AxialSeamount_Caldera.zarr', append_dim='time')