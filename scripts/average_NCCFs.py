import dask
from dask.distributed import Client, LocalCluster
import xarray as xr
import pandas as pd
import numpy as np

if __name__ == "__main__":
    dask.config.set({'temporary_directory': '/datadrive/tmp'})

    cluster = LocalCluster(n_workers=8)
    print(cluster.dashboard_link)
    client = Client(cluster)

    # Load NCCFs
    fn = '/datadrive/NCCFs/1hr_20150101_20230101_ec_cc_fcs_1_90.nc'
    NCCFs = xr.open_dataarray(fn)
    time_coord = pd.date_range(pd.Timestamp('2015-01-01'), pd.Timestamp('2022-12-31 t23:59:59.999'), freq='1H')
    NCCFs = NCCFs.assign_coords({'time':time_coord})
    NCCFs = NCCFs.chunk({'time':100, 'delay':11999})

    # Remove outliers
    print('removing outliers...')
    energy = (NCCFs*NCCFs).mean('delay')
    energy_norm = (energy/energy.std())
    threshold = 3.4
    NCCFs_or = NCCFs.where(energy_norm <= float(threshold), np.nan)

    # Average NCCFs
    NCCFs201 = NCCFs_or.rolling(time=201, center=True, min_periods=100).mean()
    NCCFs601 = NCCFs_or.rolling(time=601, center=True, min_periods=300).mean()
    NCCFs1001 = NCCFs_or.rolling(time=1001, center=True, min_periods=500).mean()

    fn = '/datadrive/NCCFs/averaged/201.nc'
    print('201 hour average...')
    NCCFs201.to_netcdf(fn)

    fn = '/datadrive/NCCFs/averaged/601.nc'
    print('601 hour average...')
    NCCFs601.to_netcdf(fn)

    fn = '/datadrive/NCCFs/averaged/1001.nc'
    print('1001 hour average...')
    NCCFs1001.to_netcdf(fn)