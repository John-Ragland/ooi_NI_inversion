import xarray as xr
from scipy import signal
from NI_tools.NI_tools import inversion, utils
from tqdm import tqdm

fbands = [[10,15], [15,20], [20,25], [25, 30], [35,40], [40, 45], [45, 50], [50,55], [55,60]]

for fband in tqdm(fbands):
    ds = xr.open_zarr('/datadrive/NCCFs/old/6year_NCCF_hilbert.zarr/')
    b,a = signal.butter(4, Wn=[fband[0]/100, fband[1]/100], btype='bandpass')
    ds_filt = utils.xr_filtfilt(ds, dim='delay', b=b, a=a)
    arrival_times = inversion.calc_prop_times(ds_filt, peaks=['dA', 's1b0A', 's1b0B', 's2b1A', 's2b1B'])
    arrival_times.to_netcdf(f'/datadrive/NCCFs/old/arrival_times_{fband[0]}_{fband[1]}Hz.nc')