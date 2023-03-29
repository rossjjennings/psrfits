import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from psrfits.dispersion import fft_roll

def symmetrize_limits(data, vmin=None, vmax=None):
    '''
    Produce symmetric limits for a set of data based on the data itself and
    (optionally) explicitly supplied upper or lower limits.
    '''
    datamin, datamax = np.nanmin(data), np.nanmax(data)
    lim = max(-datamin, datamax)
    if vmax is not None:
        lim = min(lim, vmax)
    if vmin is not None:
        lim = min(lim, -vmin)
    vmin, vmax = -lim, lim
    return vmin, vmax

def plot_portrait(ds, portrait, ax=None, sym_lim=False, vmin=None, vmax=None, phase_shift=0.5,
                  **kwargs):
    '''
    Make a pseudocolor plot of a supplied pulse portrait (pulse phase vs. frequency)
    using metadata from the given Dataset. Additional keyword arguments will be passed
    on to plt.pcolormesh().

    Parameters
    ----------
    ds:       Dataset to use
    portrait: Portrait (array of data vs. frequency and phase) to plot
    ax:       Axes in which to plot. If `None`, the current Axes will be used.
    sym_lim:  Symmetrize the colorbar limits around zero. Useful when plotting
              signed data using a diverging colormap.
    phase_shift: Fraction of a turn by which to rotate the data before plotting.
    '''
    if ax is None:
        ax = plt.gca()
    portrait = fft_roll(portrait, len(ds.phase)*phase_shift)
    if sym_lim:
        vmin, vmax = symmetrize_limits(portrait, vmin, vmax)
    phase = ds.phase - phase_shift
    freq = ds.freq.to(u.MHz).value
    pc = ax.pcolormesh(phase, freq, portrait, vmin=vmin, vmax=vmax, **kwargs)
    ax.set(xlabel='Phase (cycles)', ylabel='Frequency (MHz)')
    return pc

def plot_profile(ds, profile, ax=None, phase_shift=0.5, **kwargs):
    '''
    Make a line plot of a supplied pulse profile using metadata from the given
    Dataset. Additional keyword arguments will be passed on to plt.plot().

    Parameters
    ----------
    ds:      Dataset to use
    profile: Profile (array of data vs. pulse phase) to plot
    ax:      Axes in which to plot. If `None`, the current Axes will be used.
    phase_shift: Fraction of a turn by which to rotate the data before plotting.
    '''
    if ax is None:
        ax = plt.gca()
    profile = fft_roll(profile, len(ds.phase)*phase_shift)
    phase = ds.phase - phase_shift
    lines = ax.plot(phase, profile, **kwargs)
    ax.set_xlabel('Phase (cycles)')
    return lines

def plot_pulsetrain(ds, pulsetrain, ax=None, sym_lim=False, vmin=None, vmax=None,
                    phase_shift=0.5, **kwargs):
    '''
    Make a pseudocolor plot of a supplied "pulse train" (i.e., time series of profiles
    matching the length of the underlying data, a bit of a misnomer) using metadata
    from the given Dataset. Additional keyword arguments will be passed on to
    plt.pcolormesh().

    Parameters
    ----------
    ds:      Dataset to use
    profile: Profile (array of data vs. pulse phase) to plot
    ax:      Axes in which to plot. If `None`, the current Axes will be used.
    sym_lim: Symmetrize the colorbar limits around zero. Useful when plotting
             signed data using a diverging colormap.
    phase_shift: Fraction of a turn by which to rotate the data before plotting.
    '''
    if ax is None:
        ax = plt.gca()
    pulsetrain = fft_roll(pulsetrain, len(ds.phase)*phase_shift)
    if sym_lim:
        vmin, vmax = symmetrize_limits(pulsetrain, vmin, vmax)
    phase = ds.phase - phase_shift
    mjd = ds.epoch.mjd
    pc = ax.pcolormesh(phase, mjd, pulsetrain, vmin=vmin, vmax=vmax, **kwargs)
    ax.set(xlabel='Phase (cycles)', ylabel='MJD')
    return pc

def plot_freqtime(ds, data, ax=None, sym_lim=False, vmin=None, vmax=None, **kwargs):
    '''
    Make a pseudocolor plot of a set of data as a function of frequency and time
    using metadata from the given Dataset. Additional keyword arguments will be passed on to plt.pcolormesh().

    Parameters
    ----------
    ds:      Dataset to use
    profile: Profile (array of data vs. pulse phase) to plot
    ax:      Axes in which to plot. If `None`, the current Axes will be used.
    sym_lim: Symmetrize the colorbar limits around zero. Useful when plotting
             signed data using a diverging colormap.
    '''
    if ax is None:
        ax = plt.gca()
    if sym_lim:
        vmin, vmax = symmetrize_limits(pulsetrain, vmin, vmax)
    mjd = ds.epoch.mjd
    freq = ds.freq.to(u.MHz).value
    pc = ax.pcolormesh(mjd, freq, data.T, vmin=vmin, vmax=vmax, **kwargs)
    ax.set(xlabel='MJD', ylabel='Frequency (MHz)')
    return pc
