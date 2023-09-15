import numpy as np
import warnings

from psrfits.datafile import DataFile
from psrfits.polarization import pscrunch, to_stokes
from psrfits.dispersion import dedisperse, align_with_predictor
from psrfits.baseline import remove_baseline

def load(filename, weight=False, uniformize_freqs=False, prepare=False,
         use_predictor=True, baseline_method='avgprof', loader='memmap',
         extrap=False):
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
    loader (default: 'memmap'):
        With 'copy', the data will be read into memory immediately.
        With 'memmap' (the default), the data will be memory mapped and read into
            a Dask array, retaining a file handle.
        With 'delayed', the data will be read into a Dask array using dask.delayed.
            This is useful because it does not leave a file handle open, but is slower
            than the 'memmap' approach when the data are first read.
        With 'delayed_memmap', the data will be memory mapped and read into a Dask
            array using dask.delayed, which should not leave a file handle open.
        With 'astropy', the memory mapping internal to astropy.io.fits will be used.
    extrap (default: False): If `True`, allow extrapolation when calculating the pulse
        phase using a Tempo2 predictor. Ignored if `prepare` is `False`, or if no Tempo2
        predictor is present.
    '''
    ds = DataFile.from_file(filename, uniformize_freqs=uniformize_freqs, loader=loader)
    if prepare:
        if use_predictor:
            try:
                ds.align_with_predictor(out_of_bounds=('extrap' if extrap else 'error'))
            except AttributeError:
                ds.dedisperse()
        else:
            ds.dedisperse()
        ds.remove_baseline()
        ds.to_stokes()
    return ds
