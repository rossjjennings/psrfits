import numpy as np
import xarray as xr
from xpsrfits.polarization import get_pols

def remove_baseline(ds, method='median'):
    '''
    Remove the frequency-dependent baseline from an observation.
    '''
    if method == offpulse:
        size = ds.phase.size//8
        profile = ds.I.groupby('phase').mean()
        opw = offpulse_window(profile, size)
    
    new_data_vars = {}
    for pol in get_pols(ds):
        arr = ds.data_vars[pol]
        if method == 'median':
            baseline = arr.median(dim='phase')
        elif method == 'offpulse':
            baseline = arr.isel(phase=opw).mean()
        adjusted_arr = arr - baseline
        new_data_vars.update({pol: adjusted_arr})
    
    return xr.Dataset(new_data_vars, ds.coords, ds.attrs)

def offpulse_window(profile, size):
    '''
    Find the off-pulse window of a given profile, defined as the
    segment of pulse phase of length `size` (in phase bins)
    minimizing the integral of the pulse profile.
    '''
    bins = np.arange(len(profile))
    lower = np.argmin(rolling_sum(profile, size))
    upper = lower + size
    return lower <= bins < upper

def rolling_sum(arr, size):
    '''
    Calculate the sum of values in `arr` in a sliding window of length `size`,
    wrapping around at the end of the array.
    '''
    n = len(arr)
    s = np.cumsum(arr)
    return np.array([s[(i+size)%n]-s[i]+(i+size)//n*s[-1] for i in range(n)])
