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
import hvplot.xarray

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

ssp_m = ssp_stitch.sel({'time':pd.Timestamp('2020-03-15')}, method='nearest')
ssp_s = ssp_stitch.sel({'time':pd.Timestamp('2020-09-15')}, method='nearest')

ssp_m_interp = xr.concat((ssp_m.loc[:1523], ssp_m.loc[1524:1534].interp({'depth':np.arange(1524,1534,0.1)})), dim='depth')
ssp_s_interp = xr.concat((ssp_s.loc[:1523], ssp_s.loc[1524:1534].interp({'depth':np.arange(1524,1534,0.1)})), dim='depth')

pts_m = []
pts_s = []

del_depths = np.arange(0, 5, 0.25)

depth_cc = 1523
depth_ec = 1523
bottom_depth = 1524

del_depths_cc = []
del_depths_ec = []

for k in tqdm(range(len(del_depths))):
    for n in range(len(del_depths)):
        del_depths_cc.append(del_depths[k])
        del_depths_ec.append(del_depths[n])
        
        ssp_m = pyat_tools.convert_SSP_arlpy(ssp_m_interp, k)
        #ssp_s = pyat_tools.convert_SSP_arlpy(ssp_s_interp, k)

        fn_m = 'kraken_files/caldera_march'
        #fn_s = 'kraken_files/caldera_septe'

        pyat_tools.write_env_file_pyat(ssp_m, bottom_depth, depth_cc+del_depths[k], np.array(
            [cc_ec]), np.array([depth_ec + del_depths[n]]), 10, 'caldera_simulation_march', fn=fn_m, verbose=False)
        #pyat_tools.write_env_file_pyat(ssp_s, bottom_depth+del_depths[k], depth_cc+del_depths[k], np.array(
            #[cc_ec]), np.array([depth_ec + del_depths[k]]), 10, 'caldera_simulation_september', fn=fn_s, verbose=False)

        # Write field flp file
        s_depths = np.array([depth_cc + del_depths[k]])  # meters
        ranges = np.array([cc_ec])  # km
        r_depths = np.array([depth_ec + del_depths[n]])  # meters

        pyat_tools.write_flp_file(s_depths, ranges, r_depths, fn_m)
        #pyat_tools.write_flp_file(s_depths, ranges, r_depths, fn_s)

        data_lens = {'s_depths': len(s_depths), 'ranges': len(
            ranges), 'r_depths': len(r_depths)}
        pf_m = pyat_tools.simulate_FDGF(
            fn_m, freq_half, [0, 100], 'multiprocessing/', data_lens, True)
        #pf_s = pyat_tools.simulate_FDGF(
        #    fn_s, freq_half, [0, 100], 'multiprocessing/', data_lens, True)

        pt_m = np.real(scipy.fft.ifft(pf_m[:, 0, 0, 0]))
        #pt_s = np.real(scipy.fft.ifft(pf_s[:, 0, 0, 0]))

        pts_m.append(pt_m)
        #pts_s.append(pt_s)

results = xr.Dataset({
    'pts': xr.DataArray(np.array(pts_m), dims=['delta_depth', 'time']),
    'del_depths_cc': xr.DataArray(np.array(del_depths_cc), dims='delta_depth'),
    'del_depths_ec': xr.DataArray(np.array(del_depths_ec), dims='delta_depth')
})

fn = '/datadrive/simulation/caldera_inversion_depth_differences_seperate.nc'
results.to_netcdf(fn)

"""
# Construct DataArray from simulation and save to disc
TDGFs_march = xr.DataArray(np.array(pts_m), dims=['delta_depth', 'time'], coords={
                           'delta_depth': del_depths, 'time': t}, name='march')
#TDGFs_september = xr.DataArray(np.array(pts_s), dims=['delta_depth', 'time'], coords={
#                              'delta_depth': del_depths, 'time': t}, name='septemember')

TDGFs = xr.Dataset({'march': TDGFs_march, 'september': TDGFs_september})
TDGFs = TDGFs.assign_attrs(depth_cc=depth_cc, depth_ec=depth_ec, bottom_depth=bottom_depth)

fn = '/datadrive/simulation/caldera_inversion_depth_differences_simple.nc'

TDGFs.to_netcdf(fn)
"""