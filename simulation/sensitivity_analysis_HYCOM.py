import pyat
import pyat_tools
import xarray as xr
import pandas as pd
import numpy as np
from numpy import matlib
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm
import scipy
from geopy.distance import geodesic
from oceans.sw_extras import sw_extras as sw

## Open sound speed data
fdir = '/datadrive/HYCOM_data/Axial_Seamount_earthengine/*.nc'
ds = xr.open_mfdataset(fdir)
time_coord = pd.to_datetime(np.arange(pd.Timestamp('2015-01-01').value, pd.Timestamp('2023-01-01').value, 1e9*3600*24))
ds_interp = ds.interp(time=time_coord).compute()
ssp_full = sw.soundspeed(ds_interp.salinity, ds_interp.temperature, ds.depth)

## Write Environment File and flp files
Fs = 200
To = 30
t, freq_half, freq_full = pyat_tools.get_freq_time_vectors(Fs, To)

central_caldera = (45.954681, -130.008896)
eastern_caldera = (45.939671, -129.973801)
cc_ec = geodesic(central_caldera, eastern_caldera).km
print(cc_ec)

pts = []
for k in tqdm(range(ssp_full.shape[0])):

    ssp = pyat_tools.convert_SSP_arlpy(ssp_full, k)
    ssp = np.vstack([ssp[1:,:], np.array([1600,ssp[-1,1]])])

    fn = 'kraken_files/caldera_sensitivity_HYCOM'
    pyat_tools.write_env_file_pyat(ssp, 1499, 1498, np.array([cc_ec]), np.array(
        [1498]), 10, 'caldera_sensitivity_simulation', fn=fn, verbose=False)

    # Write field flp file
    s_depths = np.array([1498])  # meters
    ranges = np.array([cc_ec])  # km
    r_depths = np.array([1498])  # meters

    pyat_tools.write_flp_file(s_depths, ranges, r_depths, fn)

    data_lens = {'s_depths': len(s_depths), 'ranges': len(
        ranges), 'r_depths': len(r_depths)}
    
    pf = pyat_tools.simulate_FDGF(
        fn, freq_half, [0, 100], 'multiprocessing/', data_lens, True)
    pt = np.real(scipy.fft.ifft(pf[:, 0, 0, 0]))

    pts.append(pt)


# Construct DataArray from simulation and save to disc
TDGFs = xr.DataArray(np.array(pts), dims=['date', 'time'], coords={
                     'date': ssp_full.time.values, 'time': t}, name='Time Domain Greens Function')
fn = '/datadrive/simulation/caldera_sensitivity_HYCOM.nc'
TDGFs.to_netcdf(fn)