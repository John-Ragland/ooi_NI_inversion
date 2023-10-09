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
fn = '/home/jhrag/Code/ooi_NI_inversion/sampled_ssps_uniform_512.pkl'
with open(fn, 'rb') as f:
    ssps = pickle.load(f)

## Write Environment File and flp files
Fs = 200
To = 30
t, freq_half, freq_full = pyat_tools.get_freq_time_vectors(Fs, To)

central_caldera = (45.954681, -130.008896)
eastern_caldera = (45.939671, -129.973801)
cc_ec = geodesic(central_caldera, eastern_caldera).km
print(cc_ec)

pts = []
for k in tqdm(range(ssps.shape[0])):

    ssp = np.vstack((np.arange(0,1500),ssps[k,:])).T
    #ssp = np.vstack([ssp[1:,:], np.array([1600,ssp[-1,1]])])

    fn = 'kraken_files/caldera_sensitivity'
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

# write simulated PTs to disc
fn = '/datadrive/simulation/caldera_sensitivity_uniform_512.pkl'
with open(fn, 'wb') as f:
    pickle.dump(pts, f)