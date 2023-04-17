import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import hvplot.xarray
from OOI_hydrophone_cloud import utils
from OOI_hydrophone_cloud.processing import processing
import os
from dask.distributed import Client, LocalCluster
import dask


if __name__ == "__main__":
    dask.config.set({'temporary_directory': '/datadrive/tmp'})

    cluster = LocalCluster(n_workers=8)
    print(cluster.dashboard_link)
    client = Client(cluster)

    account_key = os.environ['AZURE_KEY_ooidata']
    storage_options={'account_name': 'ooidata', 'account_key': account_key}
    ds = xr.open_zarr('abfs://lfhydrophonezarr/ooi_lfhydrophones.zarr', storage_options=storage_options)

    ds_sliced = utils.slice_ds(ds, pd.Timestamp(
        '2015-01-01'), pd.Timestamp('2023-01-01'), include_coord=False)[['AXCC1', 'AXEC2']]
    NCCF_stack = processing.compute_NCCF_stack(
        ds_sliced, compute=False, stack=True)

    fn = '/datadrive/NCCFs/1hr_20150101_20230101_ec_cc.nc'
    NCCF_stack.to_netcdf(fn)
