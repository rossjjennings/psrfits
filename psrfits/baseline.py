import numpy as np
from psrfits.dataset import Dataset
from psrfits.polarization import get_pols

def remove_baseline(ds, method='offpulse'):
    '''
    Remove the frequency-dependent baseline from an observation.
    '''
    if method == 'offpulse':
        I = (ds.AA + ds.BB if ds.pol_type.startswith('AABB') else ds.I)
        size = ds.phase.size//8
        profile = I.mean(axis=(0, 1))
        opw = offpulse_window(profile, size)
    
    new_data_vars = dict(ds.data_vars)
    for pol in get_pols(ds):
        arr = ds.data_vars[pol][-1]
        if method == 'median':
            baseline = np.median(arr, axis=-1)
        elif method == 'offpulse':
            baseline = arr[:, :, opw].mean(axis=-1)
        adjusted_arr = arr - baseline[:, :, np.newaxis]
        new_data_vars.update({pol: (['time', 'freq', 'phase'], adjusted_arr)})
    
    return Dataset(new_data_vars, ds.coords, ds.attrs)

def offpulse_window(profile, size):
    '''
    Find the off-pulse window of a given profile, defined as the
    segment of pulse phase of length `size` (in phase bins)
    minimizing the integral of the pulse profile.
    '''
    bins = np.arange(len(profile))
    lower = np.argmin(rolling_sum(profile, size))
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
