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
import ODLintake

if __name__ == "__main__":
    dask.config.set({'temporary_directory': '/datadrive/tmp'})

    cluster = LocalCluster(n_workers=8)
    print(cluster.dashboard_link)
    client = Client(cluster)

    hdata = ODLintake.open_ooi_lfhydrophones()
    spec = hdata.map(xrsignal.welch, dim='time', dB=False, fs=200, nperseg=1024)
    
    spec_dB = 10*np.log10(spec/((1e-6)**2))

    spec_dB.to_netcdf('/datadrive/lfhydrophone/lfhydrophone_spectrogram.nc')