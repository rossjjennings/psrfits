import numpy as np
import warnings
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, Longitude
import astropy.units as u
import dask.array as da

from psrfits.attrs import *
from psrfits.dataset import Dataset
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

def make_dataset(filename, uniformize_freqs=False):
    '''
    Construct a Dataset object from a PSRFITS file.
    '''
    hdulist = fits.open(filename)
    primary_hdu = hdulist['primary']
    history_hdu = hdulist['history']
    subint_hdu = hdulist['subint']
    
    data = subint_hdu.data['data']
    data_vars = pol_split(data, subint_hdu.header['pol_type'])
    coords = get_coords(hdulist, uniformize_freqs)
    
    # Add data vars
    duration = subint_hdu.data['tsubint']
    data_vars['duration'] = (['time'], duration)
    
    weights = subint_hdu.data['dat_wts']
    weights = weights.reshape(coords['time'].size, coords['freq'].size)
    data_vars['weights'] = (['time', 'freq'], weights)
    
    index = subint_hdu.data['indexval']
    if not all(index == 0):
        data_vars['index'] = (['time'], index)
    
    try:
        lst = subint_hdu.data['lst_sub']
    except KeyError:
        pass
    else:
        data_vars['lst'] = (['time'], lst)
    
    try:
        ra = subint_hdu.data['ra_sub']
        dec = subint_hdu.data['dec_sub']
    except KeyError:
        pass
    else:
        if not (all(ra == ra[0]) and all(dec == dec[0])):
            data_vars['ra'] = (['time'], ra)
            data_vars['dec'] = (['time'], dec)
    
    try:
        glon = subint_hdu.data['glon_sub']
        glat = subint_hdu.data['glat_sub']
    except KeyError:
        pass
    else:
        if not (all(glon == glon[0]) and all(glat == glat[0])):
            data_vars['glon'] = (['time'], glon)
            data_vars['glat'] = (['time'], glat)
    
    try:
        feed_angle = subint_hdu.data['fd_ang']
    except KeyError:
        pass
    else:
        if not all(feed_angle == 0):
            data_vars['feed_angle'] = (['time'], feed_angle)
    
    try:
        pos_angle = subint_hdu.data['pos_ang']
    except KeyError:
        pass
    else:
        if not all(pos_angle == 0):
            data_vars['pos_angle'] = (['time'], pos_angle)
    
    try:
        par_angle = subint_hdu.data['par_ang']
    except KeyError:
        pass
    else:
        if not all(par_angle == 0):
            data_vars['par_angle'] = (['time'], par_angle)
    
    try:
        az = subint_hdu.data['tel_az']
        zen = subint_hdu.data['tel_zen']
    except KeyError:
        pass
    else:
        data_vars['az'] = (['time'], az)
        data_vars['zen'] = (['time'], zen)
    
    try:
        aux_dm = subint_hdu.data['aux_dm']
    except KeyError:
        pass
    else:
        if not all(aux_dm == 0):
            data_vars['aux_dm'] = (['time'], aux_dm)
    
    try:
        aux_rm = subint_hdu.data['aux_rm']
    except KeyError:
        pass
    else:
        if not all(aux_rm == 0):
            data_vars['aux_rm'] = (['time'], aux_rm)
    
    start_time = Time(primary_hdu.header['stt_imjd'], format='mjd')
    start_time.format = 'isot'
    start_time += primary_hdu.header['stt_smjd']*u.s
    start_time += primary_hdu.header['stt_offs']*u.s
    
    attrs = {
        'source': Source.from_hdulist(hdulist),
        'observation': Observation.from_header(primary_hdu.header),
        'telescope': Telescope.from_header(primary_hdu.header),
        'frontend': Frontend.from_header(primary_hdu.header),
        'backend': Backend.from_header(primary_hdu.header),
        'beam': Beam.from_header(primary_hdu.header),
        'calibrator': Calibrator.from_header(primary_hdu.header),
        'history': History.from_hdu(history_hdu),
        'frequency': primary_hdu.header['obsfreq'],
        'bandwidth': primary_hdu.header['obsbw'],
        'center_freq': history_hdu.data['ctr_freq'][-1],
        'channel_offset': maybe_missing(subint_hdu.header['nchnoffs']), # *
        'channel_bandwidth': subint_hdu.header['chan_bw'],
        'DM': subint_hdu.header['DM'],
        'RM': subint_hdu.header['RM'],
        'n_polns': subint_hdu.header['npol'],
        'pol_type': subint_hdu.header['pol_type'],
        'start_time': start_time,
        'start_lst': Longitude(primary_hdu.header['stt_lst']/3600, u.hourangle),
        'epoch_type': subint_hdu.header['epochs'],
        'time_var': subint_hdu.header['int_type'],
        'time_unit': subint_hdu.header['int_unit'],
        'flux_unit': subint_hdu.header['scale'],
        'time_per_bin': history_hdu.data['tbin'][-1],
        'scale': subint_hdu.data['dat_scl'],
        'offset': subint_hdu.data['dat_offs'],
    }
    
    ds = Dataset(data_vars, coords, attrs)
    
    hdulist.close()
    return ds

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

def get_coords(hdulist, uniformize_freqs):
    '''
    Get the time, frequency, and phase coordinates from a PSRFITS file.
    '''
    primary_hdu = hdulist['primary']
    history_hdu = hdulist['history']
    subint_hdu = hdulist['subint']
    
    time = subint_hdu.data['offs_sub']
    
    dat_freq = subint_hdu.data['dat_freq']
    freq = np.atleast_1d(dat_freq[0])
    channel_bandwidth = subint_hdu.header['chan_bw']
    bandwidth = primary_hdu.header['obsbw']
    if uniformize_freqs:
        # Undo effects of truncation to nearest 1 kHz (introduced by PSRCHIVE)
        freq = uniformize(freq, channel_bandwidth)
    # (Uniformized) first row should match everything to within 1 kHz
    if np.any(np.abs(dat_freq - freq) > 0.001):
        msg = 'Not all frequencies match within tolerance.'
        warnings.warn(msg, RuntimeWarning)
    # Difference between first and last frequencies should equal bandwidth
    discrepancy = np.abs(freq[-1] - freq[0] + channel_bandwidth - bandwidth)
    if discrepancy > 0.001 or (uniformize_freqs and discrepancy != 0):
        msg = 'Frequencies do not match bandwidth. Band edges may be missing.'
        warnings.warn(msg, RuntimeWarning)
    
    nbin = history_hdu.data['nbin'][-1]
    try:
        nbin_prd = int(history_hdu.data['nbin_prd'][-1])
    except ValueError:
        nbin_prd = nbin
    try:
        phs_offs = float(subint_hdu.header['phs_offs'])
    except ValueError:
        phs_offs = 0.
    phase = np.linspace(0., nbin/nbin_prd, nbin, endpoint=False) + phs_offs
    
    coords = {'time': time, 'freq': freq, 'phase': phase}
    
    return coords
