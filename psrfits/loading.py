import numpy as np
import warnings

from psrfits.datafile import DataFile
from psrfits.polarization import pscrunch, to_stokes
from psrfits.dispersion import dedisperse, align_with_predictor
from psrfits.baseline import remove_baseline

def load(filename, weight=False, uniformize_freqs=False, prepare=False,
         use_predictor=True, baseline_method='avgprof', loader='lazy'):
    '''
    Open a PSRFITS file and load the contents into a Dataset.

    Parameters
    ----------
    weight (default: False): Whether to multiply the data by the weights present in
        the file. Typically these are proportional to the time-bandwidth product.
    uniformize_freqs (default: False): Whether to replace the frequencies present in
        the file by an educated guess at a uniformly-spaced version.
    prepare (default: False): Whether to automatically dedisperse the data,
        subtract the baseline from each profile, and convert coherence data to Stokes.
        This can also be done after loading using dedicated functions.
    use_predictor (default: True): Whether to dedisperse using the Tempo2 predictor,
        if available, rather than the DM in the header. Ignored if `prepare` is `False`.
    baseline_method (default: 'avgprof'): The method used to determine the baseline
        level when it is to be removed. Options are 'avgprof', 'offpulse', and 'median'.
        See psrfits.baseline.remove_baseline() for details.
    wcfreq (default: False): Whether to use a "weighted" center frequency. This can be
        used to replicate the behaviour of PyPulse, but generally is best left alone.
    loader (default: 'lazy'): Possible values are 'lazy', 'memmap', or 'eager'. For 'lazy',
        the data will be read into a Dask array using dask.delayed, and read only when
        necessary. No file handle will be retained. For 'memmap', the data will be
        memory-mapped and read into a Dask array, retaining a file handle. For 'eager',
        the data will be read into memory immediately.
    '''
    ds = DataFile.from_file(filename, uniformize_freqs=uniformize_freqs, loader=loader)
    if prepare:
        if use_predictor:
            try:
                ds.align_with_predictor()
            except ValueError:
                ds.dedisperse()
        else:
            ds.dedisperse()
        ds.remove_baseline()
        ds.to_stokes()
    return ds
