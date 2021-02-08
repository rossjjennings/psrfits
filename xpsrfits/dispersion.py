import numpy as np
from numpy import pi, sin, cos, exp, log, sqrt
from numpy.fft import rfft, irfft, rfftfreq
import xarray as xr
from xpsrfits.dataset import Dataset
from xpsrfits.polarization import get_pols

def fft_roll(a, shift):
    '''
    Roll array by a given (possibly fractional) amount, in bins.
    Works by multiplying the FFT of the input array by exp(-2j*pi*shift*f)
    and Fourier transforming back. The sign convention matches that of 
    numpy.roll() -- positive shift is toward the end of the array.
    This is the reverse of the convention used by pypulse.utils.fftshift().
    If the array has more than one axis, the last axis is shifted.
    '''
    try:
        shift = shift[...,np.newaxis]
    except (TypeError, IndexError): pass
    phase = -2j*pi*shift*rfftfreq(a.shape[-1])
    return irfft(rfft(a)*np.exp(phase))

def weighted_center_freq(ds):
    '''
    Calculate the center frequency of the observation in the manner of PyPulse
    by weighting each channel frequency by the channel weight.
    '''
    return (np.sum(ds.freq*ds.weights)/np.sum(ds.weights)).item()

def dedisperse(ds, DM=None, weight_center_freq=False):
    '''
    Dedisperse the data with the given DM.
    If `DM` is `None`, use the DM attribute of `ds`.
    '''
    if DM is None:
        DM = ds.DM
    
    K = 1/2.41e-4
    if weight_center_freq:
        center_freq = weighted_center_freq(ds)
    else:
        center_freq = ds.center_freq
    time_delays = K*DM*(center_freq**-2 - ds.freq.values**-2)
    
    tbin = ds.time_per_bin
    bin_delays = time_delays/tbin
    
    new_data_vars = dict(ds.data_vars)
    for pol in get_pols(ds):
        dedispersed_arr = fft_roll(ds.data_vars[pol].values, bin_delays)
        new_data_vars.update({pol: (['time', 'freq', 'phase'], dedispersed_arr)})
    
    return Dataset(new_data_vars, ds.coords, ds.attrs)
