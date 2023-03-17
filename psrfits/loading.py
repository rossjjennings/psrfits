import numpy as np
import warnings
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, Longitude
import astropy.units as u
import dask.array as da

from psrfits.attrs import *
#from psrfits.dataset import Dataset
from psrfits.attrs.attrcollection import maybe_missing
from psrfits.polarization import pol_split, get_pols, pscrunch, to_stokes
from psrfits.dispersion import dedisperse, align_with_predictor
from psrfits.baseline import remove_baseline
from psrfits.uniform import uniformize

def load(filename, unpack_samples=True, weight=False, uniformize_freqs=False, prepare=False,
         use_predictor=True, baseline_method='avgprof', output_polns=None):
    '''
    Open a PSRFITS file and load the contents into a Dataset.

    Parameters
    ----------
    unpack_samples (default: True): Whether to apply the saved scale factor and
        offset to the data, converting from the on-disk integer format to 64-bit
        floating point. Disabling this can be useful for testing or inspecting the
        raw data directly.
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
    ds = make_dataset(filename, uniformize_freqs)
    if unpack_samples:
        ds = unpack(ds, weight)
    if prepare:
        if use_predictor:
            ds = align_with_predictor(ds)
        else:
            ds = dedisperse(ds)
        ds = remove_baseline(ds)
    if output_polns == 'I':
        ds = pscrunch(ds)
    elif output_polns == 'IQUV':
        ds = to_stokes(ds)
    return ds

def _load_copy(filename, hdu, column):
    '''
    Open a FITS file and make an in-memory copy of a specific HDU column.
    Intended to be used in delayed form by _load_delayed.
    '''
    hdul = fits.open(filename)
    arr = hdul[hdu].data[column].copy()
    hdul.close()
    return arr

def _load_delayed(filename, hdu, column):
    '''
    Make a single-chunk Dask array from a column of an HDU within a FITS file,
    using dask.delayed to open and read the file only when needed.
    '''
    hdul = fits.open(filename)
    shape = hdul[hdu].data[column].shape
    dtype = hdul[hdu].data[column].dtype
    hdul.close()
    arr = delayed(load_copy)(filename, hdu, column)

    # hdul[hdu].data[column] goes out of scope here and can be garbage collected.
    # If this were not the case, it would keep around an open file handle.
    return da.from_delayed(arr, shape, dtype, name=f'{filename}.{hdu}.{column}')

def _load_memmap(filename, hdu, column, chunks=None):
    '''
    Make a Dask array from a column of an HDU within a memory-mapped FITS file.
    The array will be faster to work with than if _load_delayed() were used, but
    it will keep around an open file handle, and if too many files are read
    this way, you might get a "too many open files" error.
    '''
    hdul = fits.open(filename)
    arr = hdul[hdu].data[col]
    hdul.close()
    # Providing a `name` argument to `da.from_array()` is crucial for performance,
    # since otherwise the name is derived by hashing each chunk.
    return da.from_array(arr, name=f'{filename}.{hdu}.{column}', chunks=chunks)

def unpack(ds, weight=False):
    '''
    Convert a dataset into meaningful units by apply the scaling, offset, and,
    if specified by the `weight` parameter, the weights, given in the file.
    '''
    nsub = ds.time.size
    nchan = ds.freq.size
    nbin = ds.phase.size
    npol = len(get_pols(ds))
    
    # state assumptions explicitly
    assert ds.scale.size == nsub*npol*nchan
    assert ds.offset.size == nsub*npol*nchan
    assert ds.weights.size == nsub*nchan
    scales = ds.scale.reshape(nsub, npol, nchan)
    scales = pol_split(scales, ds.pol_type)
    offsets = ds.offset.reshape(nsub, npol, nchan)
    offsets = pol_split(offsets, ds.pol_type)
    weights = ds.weights.reshape(nsub, nchan, 1)
    
    new_data_vars = dict(ds.data_vars)
    for pol in get_pols(ds):
        data = ds.data_vars[pol][-1]
        scale = np.array(scales[pol][-1]).reshape(nsub, nchan, 1)
        offset = np.array(offsets[pol][-1]).reshape(nsub, nchan, 1)
        unpacked_data = scale*data + offset
        if weight:
            unpacked_data = weights*unpacked_data
        new_data_vars[pol] = (['time', 'freq', 'phase'], unpacked_data)
    
    new_attrs = ds.attrs.copy()
    del new_attrs['scale']
    del new_attrs['offset']
    
    return Dataset(new_data_vars, ds.coords, new_attrs)
