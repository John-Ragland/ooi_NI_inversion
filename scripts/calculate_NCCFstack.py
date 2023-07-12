import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import hvplot.xarray
from OOI_hydrophone_cloud import utils
from NI_tools.NI_tools import calculate
import os
from dask.distributed import Client, LocalCluster
import dask
import ODLintake


if __name__ == "__main__":
    dask.config.set({'temporary_directory': '/datadrive/tmp'})

    cluster = LocalCluster(n_workers=8)
    print(cluster.dashboard_link)
    client = Client(cluster)

    ds = ODLintake.open_ooi_lfhydrophones()
    
    ds_sliced = utils.slice_ds(ds, pd.Timestamp(
        '2015-01-01'), pd.Timestamp('2023-01-01'), include_coord=False)[['AXCC1', 'AXEC2']]
    
    NCCF_stack = calculate.compute_NCCF_stack(
        ds_sliced, compute=False, stack=True, fcs=[1,5])

    fn = '/datadrive/NCCFs/1hr_20150101_20230101_ec_cc_fcs_1_90.nc'
    NCCF_stack.to_netcdf(fn)
