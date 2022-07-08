import numpy as np
from psrfits.dataset import Dataset
from psrfits.polarization import get_pols

def remove_baseline(ds, method='avgprof', frac=1/8):
    '''
    Remove the frequency-dependent baseline from an observation.

    Parameters
    ----------
    method (default: 'avgprof'): The method used to determine the baseline level.
        Options are 'avgprof', which takes the mean of an "off-pulse" region
        automatically determined from the average total-intensity profile,
        'offpulse', which is uses the off-pulse mean for each channel separately,
        and 'median', which takes the median of the entire profile in each channel
        separately.
    frac (default: 1/8): The fraction of the profile to use as an off-pulse window.
    '''
    if method=='avgprof':
        I = (ds.AA + ds.BB if ds.pol_type.startswith('AABB') else ds.I)
        size = int(frac*ds.phase.size)
        profile = I.mean(axis=(0, 1))
        opw = offpulse_window(profile, size)

    new_data_vars = dict(ds.data_vars)
    for pol in get_pols(ds):
        arr = ds.data_vars[pol][-1]
        if method == 'median':
            baseline = np.median(arr, axis=-1)
        elif method == 'avgprof':
            baseline = arr[..., opw].mean(axis=-1)
        elif method == 'offpulse':
            size = int(frac*ds.phase.size)
            baseline = np.min(rolling_mean(arr, size), axis=-1)
        adjusted_arr = arr - baseline[..., np.newaxis]
        new_data_vars.update({pol: (['time', 'freq', 'phase'], adjusted_arr)})
    
    return Dataset(new_data_vars, ds.coords, ds.attrs)

def offpulse_window(arr, size=None, frac=1/8):
    '''
    Find the off-pulse window of a given profile or set of profiles, defined as the
    segment of pulse phase of a given length minimizing the integral of the pulse
    profile. The length of the segment can be given as a number of pulse phase bins
    (`size`) or as a fraction of the period (`frac`). If `size` is given explicitly,
    `frac` is ignored. If a multidimensional array is given as input, the last axis
    is treated as the phase axis.
    '''
    if size is None:
        size = int(frac*arr.shape[-1])
    bins = np.arange(arr.shape[-1])
    lower = np.argmin(rolling_mean(arr, size), axis=-1)
    upper = lower + size
    try:
        lower = lower[..., np.newaxis]
        upper = upper[..., np.newaxis]
    except (TypeError, IndexError):
        pass
    return np.logical_and(lower <= bins, bins < upper)

def rolling_mean(arr, size):
    '''
    Calculate the mean of values in `arr` in a sliding window of length `size`,
    wrapping around at the end of the array. If `arr` is more than one-dimensional,
    the rolling mean will be computed over the last dimension only.
    '''
    n = arr.shape[-1]
    filtr = np.zeros(n)
    filtr[-size:] = 1
    return np.fft.irfft(np.fft.rfft(arr)*np.fft.rfft(filtr))/size

def offpulse_rms(profile, size=None, frac=1/8):
    '''
    Calculate the off-pulse RMS of a profile (a measure of noise level).
    This is the RMS of `profile` in the segment of length `size`
    (in phase bins) minimizing the integral of `profile`.
    '''
    opw = offpulse_window(profile, size, frac)
    return np.sqrt(np.mean(profile[opw]**2, axis=-1))
