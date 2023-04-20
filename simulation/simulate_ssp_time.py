import pyat
from pyat_tools import pyat_tools
import xarray as xr
import pandas as pd
import numpy as np
from numpy import matlib
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm
import scipy
from geopy.distance import geodesic

## Load Stitched SSP
fn = '/datadrive/CTD_data/stitched_2015_2023_ooipy.nc'
ssp_stitch = xr.open_dataarray(fn)

## Write Environment File and flp files
Fs = 200
To = 30
t, freq_half, freq_full = pyat_tools.get_freq_time_vectors(Fs, To)

central_caldera = (45.954681, -130.008896)
eastern_caldera = (45.939671, -129.973801)
cc_ec = geodesic(central_caldera, eastern_caldera).km

pts = []
for k in tqdm(range(ssp_stitch.shape[0])):

    ssp = pyat_tools.convert_SSP_arlpy(ssp_stitch, k)
    fn = 'kraken_files/caldera'
    pyat_tools.write_env_file_pyat(ssp, 1529, 1528, np.array([cc_ec]), np.array(
        [1519]), 10, 'caldera_simulation', fn=fn, verbose=False)

    # Write field flp file
    s_depths = np.array([1528])  # meters
    ranges = np.array([cc_ec])  # km
    r_depths = np.array([1519])  # meters

    pyat_tools.write_flp_file(s_depths, ranges, r_depths, fn)

    data_lens = {'s_depths': len(s_depths), 'ranges': len(
        ranges), 'r_depths': len(r_depths)}
    pf = pyat_tools.simulate_FDGF(
        fn, freq_half, [0, 100], 'multiprocessing/', data_lens, True)
    pt = np.real(scipy.fft.ifft(pf[:, 0, 0, 0]))
    pts.append(pt)

# Construct DataArray from simulation and save to disc
TDGFs = xr.DataArray(np.array(pts), dims=['date', 'time'], coords={
                     'date': ssp_stitch.time.values, 'time': t}, name='Time Domain Greens Function')
fn = '/datadrive/simulation/caldera_inversion_timeseries.nc'
TDGFs.to_netcdf(fn)
