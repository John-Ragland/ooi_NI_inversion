import xarray as xr
import numpy as np
from OOI_hydrophone_cloud import utils
from xrsignal import xrsignal
from scipy import signal
from tqdm import tqdm

fbands = [[1,5], [5, 10], [10,15], [15,20], [20,25], [25, 30], [35,40], [40, 45], [45, 50], [50,55], [55,60]]

ds = xr.open_dataset('/datadrive/NCCFs/old/6year_NCCF_201.nc')
ds = ds.chunk({'delay':11999, 'dates':50})

for fband in tqdm(fbands):
    b,a = signal.butter(4, [fband[0]/100, fband[1]/100], btype='bandpass')
    ds_filt = xrsignal.filtfilt(ds, dim='delay', b=b, a=a).compute()
    fn = f'/datadrive/NCCFs/old/filtered/{fband[0]}-{fband[1]}.nc'
    ds_filt.to_netcdf(fn)