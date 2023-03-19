import numpy as np
import warnings
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, Longitude
import astropy.units as u
import dask.array as da
import warnings

from psrfits.attrs import *
from psrfits.dataset import DataFile
from psrfits.attrs.attrcollection import maybe_missing
from psrfits.polarization import get_pols, pscrunch, to_stokes
from psrfits.dispersion import dedisperse, align_with_predictor
from psrfits.baseline import remove_baseline
from psrfits.uniform import uniformize

def load(filename, weight=False, uniformize_freqs=False, prepare=False,
         use_predictor=True, baseline_method='avgprof', output_polns=None):
    '''
    Open a PSRFITS file and load the contents into a Dataset.

    Parameters
    ----------
    weight (default: False): Whether to multiply the data by the weights present in
        the file. Typically these are proportional to the time-bandwidth product.
    uniformize_freqs (default: False): Whether to replace the frequencies present in
        the file by an educated guess at a uniformly-spaced version.
    prepare (default: False): Whether to automatically dedisperse the data and
        subtract the baseline from each profile. This can also be done after loading
        using dedicated functions.
    use_predictor (default: True): Whether to dedisperse using the Tempo2 predictor,
        if available, rather than the DM in the header. Ignored if `prepare` is `False`.
    baseline_method (default: 'avgprof'): The method used to determine the baseline
        level when it is to be removed. Options are 'avgprof', 'offpulse', and 'median'.
        See psrfits.baseline.remove_baseline() for details.
    wcfreq (default: False): Whether to use a "weighted" center frequency. This can be
        used to replicate the behaviour of PyPulse, but generally is best left alone.
    output_polns (default: None): Output polarizations to produce. The value 'IQUV'
        will produce all four Stokes parameters; a value of 'I' gives intensity only.
        Any other value will leave the polarizations in the file unchanged.
    '''
    ds = DataFile.from_file(filename, uniformize_freqs=uniformize_freqs)
    if prepare:
        if use_predictor:
            try:
                ds.align_with_predictor()
            except ValueError:
                warnings.warn('No Tempo2 predictor present! Using fallback dedispersion method')
                ds.dedisperse()
        else:
            ds.dedisperse()
        ds.remove_baseline()
    if output_polns == 'I':
        ds.pscrunch()
    elif output_polns == 'IQUV':
        ds.to_stokes()
    return ds
