import numpy as np
from numpy import pi, sin, cos, exp, log, sqrt
from numpy.fft import rfft, irfft, rfftfreq
import astropy.units as u
from psrfits.polarization import get_pols
import dask.array as da

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
    K = 1/2.41e-4 * u.s * u.MHz**2 * u.cm**3 / u.pc

    if weight_center_freq:
        center_freq = weighted_center_freq(ds)
    else:
        center_freq = ds.history.center_freq

    return (K*DM*(ds.freq**-2 - center_freq**-2)).to(u.s)

def dedisperse(ds, inplace=False, DM=None, weight_center_freq=False):
    '''
    Dedisperse the data with the given DM.
    If `DM` is `None`, use the DM attribute of `ds`.
    '''
    time_delays = dispersion_dt(ds, DM, weight_center_freq)
    
    tbin = ds.history.time_per_bin
    bin_delays = time_delays/tbin

    new_ds = ds if inplace else ds.copy()
    for pol in get_pols(ds):
        dedispersed_arr = fft_roll(getattr(ds, pol), -bin_delays)
        setattr(new_ds, pol, dedispersed_arr)
    
    return new_ds

def channel_phase(ds, out_of_bounds='error'):
    '''
    Compute the phase offset in each channel from the internal Tempo2 predictor.
    '''
    if ds.source.predictor is None:
        raise ValueError("No Tempo2 predictor present!")
    phase_offs = ds.source.predictor(ds.epoch[:, np.newaxis], ds.freq, out_of_bounds) % 1
    return phase_offs

def align_with_predictor(ds, inplace=False, out_of_bounds='error'):
    '''
    Dedisperse and align the data using the internal Tempo2 predictor.
    '''
    phase_offs = np.float64(channel_phase(ds, out_of_bounds))

    # Converting phase_offs to a Dask array first speeds this up significantly. IDK why...
    if any(isinstance(getattr(ds, pol), da.Array) for pol in get_pols(ds)):
        phase_offs = da.from_array(phase_offs)

    new_ds = ds if inplace else ds.copy()
    for pol in get_pols(ds):
        pol_arr = getattr(ds, pol)
        nbin = pol_arr.shape[-1]
        dedispersed_arr = fft_roll(pol_arr, nbin*phase_offs)
        setattr(new_ds, pol, dedispersed_arr)

    return new_ds
