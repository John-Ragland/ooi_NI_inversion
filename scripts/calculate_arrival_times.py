from dask.distributed import LocalCluster, Client
import dask
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dask.distributed import Client
from xrsignal import xrsignal
from scipy import signal
from NI_tools.NI_tools import inversion
import hvplot.xarray
import holoviews as hv
from tqdm import tqdm

if __name__ == "__main__":
    dask.config.set({'temporary_directory': '/datadrive/tmp'})
    
    cluster = LocalCluster(n_workers=8)
    print(cluster.dashboard_link)
    client = Client(cluster)

    for threshold in tqdm([3,4,5,100]):
        fn = '/datadrive/NCCFs/1hr_20150101_20230101_ec_cc.nc'
        NCCFs = xr.open_dataarray(fn).chunk({'delay':11999, 'time':100})
        NCCFs = NCCFs.rename({'time':'dates'})

        time_coord = pd.to_datetime(np.arange(pd.Timestamp('2015-01-01').value, pd.Timestamp('2023-01-01').value, 1e9*3600))
        NCCFs = NCCFs.assign_coords({'dates':time_coord})

        ## Remove outliers
        energy = (NCCFs*NCCFs).mean('delay').compute()
        energy_norm = (energy)/energy.std()

        mask = energy_norm <= threshold
        NCCFs[~mask,:] = np.nan

        ## Compute arrival times 
        NCCFs201 = NCCFs.rolling(dates=201, min_periods=100, center=True).mean()
        NCCFs601 = NCCFs.rolling(dates=601, min_periods=100, center=True).mean()
        NCCFs1001 = NCCFs.rolling(dates=1001, min_periods=100, center=True).mean()

        NCCFs_avg = [NCCFs201, NCCFs601, NCCFs1001]

        # Filter NCCFs
        fcs = [
            [46.45346687092607, 35.528258201431036],
            [54.724490650555055, 38.998664801096524],
            [60.603626021214424, 40.2010426874647]
        ]

        bs_A = []
        as_A = []
        bs_B = []
        as_B = []

        for fc in fcs:
            b,a = signal.butter(4, Wn=[fc[0]/100], btype='lowpass')
            bs_A.append(b)
            as_A.append(a)
            b,a = signal.butter(4, Wn=[fc[1]/100], btype='lowpass')
            bs_B.append(b)
            as_B.append(a)

        filtered_NCCFs_A = []
        filtered_NCCFs_B = []

        avg_hours = [201, 601, 1001]

        for k in range(3):
            filtered_NCCFs_A.append(xrsignal.hilbert_mag(xrsignal.filtfilt(NCCFs_avg[k], dim='delay', b=bs_A[k], a=as_A[k]), dim='delay'))
            filtered_NCCFs_A[-1].attrs = {'fc':fcs[k][0], 'avg_hours':avg_hours[k]}
            filtered_NCCFs_B.append(xrsignal.hilbert_mag(xrsignal.filtfilt(NCCFs_avg[k], dim='delay', b=bs_B[k], a=as_B[k]), dim='delay'))
            filtered_NCCFs_B[-1].attrs = {'fc':fcs[k][1], 'avg_hours':avg_hours[k]}

        ats = []
        for k in range(3):
            at = inversion.calc_prop_times(xr.Dataset({'NCCFs':filtered_NCCFs_A[k]}), peaks=['s1b0A'])['NCCFs'].loc['s1b0A',:]
            ats.append(at)
            
            at = inversion.calc_prop_times(xr.Dataset({'NCCFs':filtered_NCCFs_B[k]}), peaks=['s1b0B'])['NCCFs'].loc['s1b0B',:]
            ats.append(at)

        ats_x = xr.concat(ats, dim='avg_hour').assign_coords({'avg_hour':[201, 201, 601, 601, 1001, 1001]})

        if threshold == 100:
            fn = '/datadrive/NCCFs/arrival_times/no_outlier_removal.nc'
        else:
            fn = f'/datadrive/NCCFs/arrival_times/outlier_removal_threshold_{threshold}.nc'
        ats_x.to_netcdf(fn)
    
