import numpy as np
from dask.diagnostics import ProgressBar

def avg_portrait(ds, poln='I', start_mjd=-np.inf, stop_mjd=np.inf, use_weights=True,
                 compute=True, progress=False):
    '''
    Calculate an average portrait (frequency vs. pulse phase) from the data.

    Parameters
    ----------
    ds: Dataset to use
    poln: Polarization to use
    start_mjd: Initial time, as an MJD. Data before this point will be excluded.
    stop_mjd: Final time, as an MJD. Data after this point will be excluded.
    use_weights: Compute a weighted average. If `True`, the weights stored in the
                 Dataset will be used. If `False`, no weights will be used.
    compute: Call compute() on the result, if it is a Dask array.
    progress: Display a progress bar during the computation.
    '''
    valid_mjds = (ds.epoch.mjd >= start_mjd) & (ds.epoch.mjd < stop_mjd)
    data = getattr(ds, poln)[valid_mjds]
    if use_weights:
        weights = ds.weights[valid_mjds, :, np.newaxis]
        portrait = np.nanmean(weights*data, axis=0)
        portrait /= np.nanmean(weights, axis=0)
    else:
        portrait = np.nanmean(data)

    if hasattr(portrait, 'compute') and compute:
        if progress:
            with ProgressBar():
                portrait = portrait.compute()
        else:
            portrait = portrait.compute()
    return portrait

def avg_profile(ds, poln='I', low_freq=-np.inf, high_freq=np.inf, start_mjd=-np.inf,
                stop_mjd=np.inf, use_weights=True, compute=True, progress=False):
    '''
    Calculate an average profile from the data.

    Parameters
    ----------
    ds: Dataset to use
    poln: Polarization to use
    low_freq: Low frequency, as a Quantity. Lower-frequency data will be excluded.
    high_freq: High frequency, as a Quantity. Higher-frequency data will be excluded.
    start_mjd: Initial time, as an MJD. Data before this point will be excluded.
    stop_mjd: Final time, as an MJD. Data after this point will be excluded.
    use_weights: Compute a weighted average. If `True`, the weights stored in the
                 Dataset will be used. If `False`, no weights will be used.
    compute: Call compute() on the result, if it is a Dask array.
    progress: Display a progress bar during the computation.
    '''
    valid_mjds = (ds.epoch.mjd >= start_mjd) & (ds.epoch.mjd < stop_mjd)
    valid_freqs = (ds.freq >= low_freq) & (ds.freq < high_freq)

    # appease Dask by only fancy indexing on one axis at a time
    data = getattr(ds, poln)[valid_mjds]
    data = data[:, valid_freqs]
    if use_weights:
        weights = ds.weights[valid_mjds, :, np.newaxis]
        weights = weights[:, valid_freqs]
        profile = np.nanmean(weights*data, axis=(0, 1))
        profile /= np.nanmean(weights, axis=(0, 1))
    else:
        profile = np.nanmean(data, axis=(0, 1))

    if hasattr(profile, 'compute') and compute:
        if progress:
            with ProgressBar():
                profile = profile.compute()
        else:
            profile = profile.compute()
    return profile

def avg_pulsetrain(ds, poln='I', low_freq=-np.inf, high_freq=np.inf, use_weights=True,
                   compute=True, progress=False):
    '''
    Calculate an average pulse train (time vs. pulse phase, a bit of a misnomer)
    from the data.

    Parameters
    ----------
    ds: Dataset to use
    poln: Polarization to use
    low_freq: Low frequency, as a Quantity. Lower-frequency data will be excluded.
    high_freq: High frequency, as a Quantity. Higher-frequency data will be excluded.
    use_weights: Compute a weighted average. If `True`, the weights stored in the
                 Dataset will be used. If `False`, no weights will be used.
    compute: Call compute() on the result, if it is a Dask array.
    progress: Display a progress bar during the computation.
    '''
    valid_freqs = (ds.freq >= low_freq) & (ds.freq < high_freq)
    data = getattr(ds, poln)[:, valid_freqs]
    if use_weights:
        weights = ds.weights[:, valid_freqs, np.newaxis]
        pulsetrain = np.nanmean(weights*data, axis=1)
        pulsetrain /= np.nanmean(weights, axis=1)
    else:
        pulsetrain = np.nanmean(data, axis=1)

    if hasattr(pulsetrain, 'compute') and compute:
        if progress:
            with ProgressBar():
                pulsetrain = pulsetrain.compute()
        else:
            pulsetrain = pulsetrain.compute()
    return pulsetrain
