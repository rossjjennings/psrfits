import numpy as np
import xarray as xr
import warnings
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, Longitude
import astropy.units as u
from xpsrfits.attrs import *
from xpsrfits.dataset import Dataset
from xpsrfits.attrs.attrcollection import maybe_missing
from xpsrfits.polarization import pol_split, get_pols, pscrunch, to_stokes
from xpsrfits.dispersion import dedisperse
from xpsrfits.baseline import remove_baseline
from xpsrfits.uniform import uniformize

def ingest(filename, weight=False, DM=None, wcfreq=False,
           baseline_method='offpulse', output_polns='IQUV'):
    '''
    Load a PSRFITS file, dedisperse and remove the baseline.
    '''
    ds = load(filename, weight)
    ds = dedisperse(ds, DM, weight_center_freq=wcfreq)
    ds = remove_baseline(ds, method=baseline_method)
    if output_polns == 'I':
        ds = pscrunch(ds)
    elif output_polns == 'IQUV':
        ds = to_stokes(ds)
    return ds

def read(filename):
    '''
    Open a PSRFITS file and load the contents into an xarray Dataset.
    '''
    with fits.open(filename) as hdulist:
        ds = to_dataset(hdulist)
    return ds

def load(filename, weight=False, uniformize_freqs=True):
    with fits.open(filename) as hdulist:
        ds = to_dataset(hdulist, uniformize_freqs)
    ds = unpack(ds, weight)
    return ds

def to_dataset(hdulist, uniformize_freqs=False):
    '''
    Convert a FITS HDUList object into an xarray Dataset.
    '''
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
            data_vars['glon'] = (['time'], glat)
    
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
    weights = ds.weights.data.reshape(nsub, nchan, 1)
    
    new_data_vars = dict(ds.data_vars)
    for pol in get_pols(ds):
        data = ds.data_vars[pol].data
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
