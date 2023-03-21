import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

def sym_lim(data, vmin=None, vmax=None):
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

def plot_portrait(ds, portrait, ax=None, sym_lim=False, vmin=None, vmax=None, **kwargs):
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
    '''
    if ax is None:
        ax = plt.gca()
    portrait = np.roll(portrait, len(ds.phase)//2, axis=-1)
    if sym_lim:
        vmin, vmax = sym_lim(portrait, vmin, vmax)
    phase = ds.phase - ds.phase[len(ds.phase)//2]
    freq = ds.freq.to(u.MHz).value
    pc = ax.pcolormesh(phase, freq, portrait, vmin=vmin, vmax=vmax, **kwargs)
    ax.set(xlabel='Phase (cycles)', ylabel='Frequency (MHz)')
    return pc

def plot_profile(ds, profile, ax=None, **kwargs):
    '''
    Make a line plot of a supplied pulse profile using metadata from the given
    Dataset. Additional keyword arguments will be passed on to plt.plot().

    Parameters
    ----------
    ds:      Dataset to use
    profile: Profile (array of data vs. pulse phase) to plot
    ax:      Axes in which to plot. If `None`, the current Axes will be used.
    '''
    if ax is None:
        ax = plt.gca()
    profile = np.roll(profile, len(ds.phase)//2)
    phase = ds.phase - ds.phase[len(ds.phase)//2]
    lines = ax.plot(phase, profile, **kwargs)
    ax.set_xlabel('Phase (cycles)')
    return lines

def plot_pulsetrain(ds, pulsetrain, ax=None, sym_lim=False, vmin=None, vmax=None, **kwargs):
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
    '''
    if ax is None:
        ax = plt.gca()
    pulsetrain = np.roll(pulsetrain, len(ds.phase)//2, axis=-1)
    if sym_lim:
        vmin, vmax = sym_lim(pulsetrain, vmin, vmax)
    phase = ds.phase - ds.phase[len(ds.phase)//2]
    mjd = ds.epoch.mjd
    pc = ax.pcolormesh(phase, mjd, pulsetrain, vmin=vmin, vmax=vmax, **kwargs)
    ax.set(xlabel='Phase (cycles)', ylabel='MJD')
    return pc
