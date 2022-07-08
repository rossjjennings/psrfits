import numpy as np
from numpy import pi, sin, cos, exp, log, sqrt
from numpy.fft import rfft, irfft, rfftfreq
import astropy.units as u
from psrfits.dataset import Dataset
from psrfits.polarization import get_pols

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

def dispersion_dt(ds, DM=None, weight_center_freq=False):
    '''
    Compute the time delay in each channel associated with a given DM.
    If `DM` is `None`, use the DM attribute of `ds`.
    '''
    if DM is None:
        DM = ds.DM

    # TEMPO/Tempo2/PRESTO conventional value (cf. Kulkarni 2020, arXiv:2007.02886)
    K = 1/2.41e-4 # s MHz**2 cm**3 / pc

    if weight_center_freq:
        center_freq = weighted_center_freq(ds)
    else:
        center_freq = ds.center_freq

    return K*DM*(ds.freq**-2 - center_freq**-2)

def dedisperse(ds, DM=None, weight_center_freq=False):
    '''
    Dedisperse the data with the given DM.
    If `DM` is `None`, use the DM attribute of `ds`.
    '''
    time_delays = dispersion_dt(ds, DM, weight_center_freq)
    
    tbin = ds.time_per_bin
    bin_delays = time_delays/tbin
    
    new_data_vars = dict(ds.data_vars)
    for pol in get_pols(ds):
        dedispersed_arr = fft_roll(ds.data_vars[pol][-1], -bin_delays)
        new_data_vars.update({pol: (['time', 'freq', 'phase'], dedispersed_arr)})
    
    return Dataset(new_data_vars, ds.coords, ds.attrs)

def channel_phase(ds, extrap=False):
    '''
    Compute the phase offset in each channel from the internal Tempo2 predictor.
    '''
    if ds.source.predictor is None:
        raise ValueError("No Tempo2 predictor present!")
    phase_offs = np.empty_like(ds.freq)
    epoch = ds.start_time + ds.time[0]*u.s
    for i, f in enumerate(ds.freq):
        phase_offs[i] = ds.source.predictor(epoch.mjd_long, f, extrap) % 1
    return phase_offs

def align_with_predictor(ds, extrap=False):
    '''
    Dedisperse and align the data using the internal Tempo2 predictor.
    '''
    phase_offs = channel_phase(ds)

    new_data_vars = dict(ds.data_vars)
    for pol in get_pols(ds):
        pol_arr = ds.data_vars[pol][-1]
        nbin = pol_arr.shape[-1]
        dedispersed_arr = fft_roll(pol_arr, nbin*phase_offs)
        new_data_vars.update({pol: (['time', 'freq', 'phase'], dedispersed_arr)})

    return Dataset(new_data_vars, ds.coords, ds.attrs)
