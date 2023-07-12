import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dask.distributed import Client
from tqdm import tqdm
from xrsignal import xrsignal
import scipy
from scipy import signal
from functools import partial
import xrft

fn = '/datadrive/NCCFs/averaged/201.nc'
NCCFs201 = xr.open_dataarray(fn).chunk({'time':37*3})
s1b0A = NCCFs201.sel({'delay':slice(-3.5, -2.5)})

s1b0A_Sxx = xrsignal.welch(s1b0A, dim='delay', return_onesided=False, nperseg=200, fs=200)[:,:,0]
s1b0A_Sxx = s1b0A_Sxx.rename({'delay_frequency':'frequency'}).assign_coords({'time':s1b0A.time})

s1b0A_f = xrft.fft(s1b0A, dim='delay').rename({'freq_delay':'frequency'})

s1b0A_match_f = s1b0A_Sxx.drop(['time', 'frequency']) * s1b0A_f.drop(['time', 'frequency'])
s1b0A_match_f = s1b0A_match_f.assign_coords(s1b0A_Sxx.coords)

s1b0A_match = xrsignal.hilbert_mag(xrft.ifft(s1b0A_match_f, dim='frequency').rename({'freq_frequency':'delay'}).real, dim='delay')
s1b0A_match = s1b0A_match.assign_coords({'delay':s1b0A.delay})

s1b0A_match_us = s1b0A_match.interp(delay=np.arange(-3.1, -2.9, 0.000001))