import xarray as xr
import hvplot.xarray
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

# open dataset
fn = '/datadrive/CTD_data/data_explorer/ctd_data.nc'
ds = xr.load_dataset(fn)

# remove duplicates
duplicate_mask = np.diff(ds['time']) == np.array(0, dtype='timedelta64[ns]')
duplicate_mask = np.concatenate((np.array([False]), duplicate_mask))

ds_nodup = ds.isel({'row':~duplicate_mask})

ds_starttime = pd.Timestamp(ds_nodup['time'][0].values)
ds_endtime = pd.Timestamp(ds_nodup['time'][-1].values)

first_day = pd.Timestamp(ds_starttime.strftime('%Y-%m-%d'))
time_coord_day = pd.to_datetime(np.arange(first_day.value, ds_endtime.value, 3600*24*1e9,))

# Construct Xarray.DataArrays
var_names = list(ds.keys())
var_names.remove('time')
var_names.remove('z')

das_grid = []
depth_coord = np.arange(0,-200, -1)
for var_name in var_names:
      das_grid.append(xr.DataArray(np.ones((len(time_coord_day), 200))*np.nan, dims=['time','depth'], coords={'depth':depth_coord, 'time':time_coord_day}, name=var_name))

ds_grid = xr.Dataset(dict(zip(var_names, das_grid)))

# slice dataset for single day
start_time = pd.Timestamp('2017-01-01')
end_time = pd.Timestamp('2017-01-02')

for i in tqdm(range(len(time_coord_day)), position=0):

    # Slice dataset to specific day
    start_time = time_coord_day[i]
    end_time = start_time + pd.Timedelta(days=1)
    time_mask = (ds_nodup.time.values > start_time) & (ds_nodup.time.values < end_time)
    time_ids = np.where(time_mask)[0]
    ds_day = ds_nodup.isel({'row':time_ids})

    # loop through depth
    for z in depth_coord:
        depth_idxs = np.where((ds_day.z == z).values)[0]
        ds_single = ds_day.isel({'row':depth_idxs}).mean()
        
        for var_name in var_names:
            ds_grid[var_name].loc[{'depth':z, 'time':start_time}] = ds_single[var_name].values

ds_grid.to_netcdf('/datadrive/CTD_data/data_explorer/gridded_CTD.nc')