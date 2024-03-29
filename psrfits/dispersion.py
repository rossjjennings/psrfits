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

def dispersion_dt(ds, DM=None, ref_freq=None):
    '''
    Compute the time delay in each channel associated with a given DM
    and reference frequency. If `DM` is `None`, use the DM attribute of `ds`.
    If `ref_freq` is `None`, use the polyco reference frequency.
    '''
    if DM is None:
        DM = ds.DM

    # TEMPO/Tempo2/PRESTO conventional value (cf. Kulkarni 2020, arXiv:2007.02886)
    K = 1/2.41e-4 * u.s * u.MHz**2 * u.cm**3 / u.pc

    if ref_freq is None:
        ref_freq = ds.polyco.ref_freq
    elif ref_freq == 'weighted':
        ref_freq = weighted_center_freq(ds)

    return (K*DM*(ds.freq**-2 - ref_freq**-2)).to(u.s)

def dedisperse(ds, inplace=False, DM=None, ref_freq=None):
    '''
    Dedisperse the data with the given DM and reference frequency.
    If the data is already dedispersed, dedisperse to the new DM.
    If `DM` is `None`, use the DM attribute of `ds`, and do nothing
    if the data is already dedispersed.
    If `ref_freq` is `None`, use the polyco reference frequency.
    '''
    if DM is None and ds.history.dedispersed:
        return ds
    elif ds.history.dedispersed:
        delta_DM = DM - ds.history.DM
    else:
        delta_DM = DM

    time_delays = dispersion_dt(ds, delta_DM, ref_freq)

    tbin = ds.history.time_per_bin
    bin_delays = (time_delays/tbin).to("").value

    new_ds = ds if inplace else ds.copy()
    for pol in get_pols(ds):
        dedispersed_arr = fft_roll(getattr(ds, pol), -bin_delays)
        setattr(new_ds, pol, dedispersed_arr)

    dedisp_method='Polyco' if ref_freq is None else f'Incoherent(ref_freq={ref_freq})'
    if DM is None:
        DM = ds.DM
    new_ds.DM = DM
    new_ds.history.add_entry(dedispersed=True, DM=DM, dedisp_method='Polyco')

    return new_ds

def channel_phase(ds, out_of_bounds='error'):
    '''
    Compute the phase offset in each channel from the internal Tempo2 predictor.
    '''
    if ds.predictor is None:
        raise ValueError("No Tempo2 predictor present!")
    phase_offs = ds.predictor(ds.epoch[:, np.newaxis], ds.freq, out_of_bounds) % 1
    return phase_offs

def align_with_predictor(ds, inplace=False, out_of_bounds='error'):
    '''
    Dedisperse and align the data using the internal Tempo2 predictor.
    If the data is already dedispersed, do nothing.
    '''
    if ds.history.dedispersed:
        return ds

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

    new_ds.history.add_entry(dedispersed=True, dedisp_method='T2Predict')

    return new_ds
