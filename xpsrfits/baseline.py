import numpy as np
import xarray as xr
from xpsrfits.dataset import Dataset
from xpsrfits.polarization import get_pols

def remove_baseline(ds, method='median'):
    '''
    Remove the frequency-dependent baseline from an observation.
    '''
    if method == 'offpulse':
        I = (ds.AA + ds.BB if ds.pol_type.startswith('AABB') else ds.I)
        size = ds.phase.size//8
        profile = I.groupby('phase').mean(dim=...)
        opw = offpulse_window(profile, size)
    
    new_data_vars = dict(ds.data_vars)
    for pol in get_pols(ds):
        arr = ds.data_vars[pol]
        if method == 'median':
            baseline = arr.median(dim='phase')
        elif method == 'offpulse':
            baseline = arr.isel(phase=opw).mean(dim='phase')
        adjusted_arr = arr - baseline
        new_data_vars.update({pol: adjusted_arr})
    
    return Dataset(new_data_vars, ds.coords, ds.attrs)

def offpulse_window(profile, size):
    '''
    Find the off-pulse window of a given profile, defined as the
    segment of pulse phase of length `size` (in phase bins)
    minimizing the integral of the pulse profile.
    '''
    bins = np.arange(len(profile))
    lower = np.argmin(rolling_sum(profile.values, size))
    upper = lower + size
    return np.logical_and(lower <= bins, bins < upper)

def rolling_sum(arr, size):
    '''
    Calculate the sum of values in `arr` in a sliding window of length `size`,
    wrapping around at the end of the array.
    '''
    n = len(arr)
    s = np.cumsum(arr)
    return np.array([s[(i+size)%n]-s[i]+(i+size)//n*s[-1] for i in range(n)])

def offpulse_rms(profile, size):
    '''
    Calculate the off-pulse RMS of a profile (a measure of noise level).
    This is the RMS of `profile` in the segment of length `size`
    (in phase bins) minimizing the integral of `profile`.
    '''
    opw = offpulse_window(profile)
    return np.sqrt(profile.isel(phase=opw).mean(dim='phase'))
