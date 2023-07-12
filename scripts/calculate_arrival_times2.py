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

    peaks = ['dA', 's1b0A', 's1b0B', 's2b1A', 's2b1B']

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

        NCCFs_avg = {
            '201':NCCFs201,
            '601':NCCFs601,
            '1001':NCCFs1001
        }

        # coherency frequency cutoffs for different peaks
        #   these are calculated using bilinear model and peak spectrums 
        fcs = {
            '201': {'dA': 38.49644799876388, 's1b0A': 46.45346687092607, 's1b0B': 35.528258201431036, 's2b1A': 31.60147483493909, 's2b1B': 18.426651669398172},
            '601': {'dA': 40.55017262364214, 's1b0A': 54.724490650555055, 's1b0B': 38.998664801096524, 's2b1A': 34.214072643236165, 's2b1B': 24.394969444999766},
            '1001': {'dA': 42.36289799469905, 's1b0A': 60.603626021214424, 's1b0B': 40.2010426874647, 's2b1A': 35.17584754198018, 's2b1B': 26.25266444094792}
            }

        ats = {
            '201': {},
            '601': {},
            '1001': {}
        }

        # compute arrival times for each peak and averaging time
        for avg_hour in fcs.keys():
            
            
            for peak in fcs[avg_hour].keys():
                b,a = signal.butter(4, Wn=[fcs[avg_hour][peak]/100], btype='lowpass')

                NCCFs_filt = xrsignal.filtfilt(NCCFs_avg[avg_hour], dim='delay', b=b, a=a)
                NCCFs_c = xrsignal.hilbert_mag(NCCFs_filt, dim='delay')

                at = inversion.calc_prop_times(xr.Dataset({'NCCFs':NCCFs_c}), peaks=[peak])['NCCFs'].loc[peak,:].drop_vars('peak')
                at.attrs = {'avg_hour':avg_hour, 'fc':fcs[avg_hour][peak]}
                
                ats[avg_hour][peak] = at


        for avg_hour in ats.keys():
            atx = xr.Dataset(ats[avg_hour])

            fn= f'/datadrive/NCCFs/arrival_times/outlier{threshold}_avghour{avg_hour}.nc'
            atx = atx.to_netcdf(fn)
    
