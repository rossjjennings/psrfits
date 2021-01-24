import numpy as np
import xarray as xr
import warnings
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from pint.models import get_model
import tempfile
from xpsrfits.attrs import *
from xpsrfits.polarization import pol_split, pscrunch, to_stokes
from xpsrfits.dispersion import dedisperse
from xpsrfits.baseline import remove_baseline

def ingest(filename, weight=True, DM=None, wcfreq=False,
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

def load(filename, weight=True):
    '''
    Open a PSRFITS file and load the contents into an xarray Dataset.
    '''
    with fits.open(filename) as hdulist:
        ds = to_dataset(hdulist, weight)
    return ds

def to_dataset(hdulist, weight=True): 
    '''
    Convert a FITS HDUList object into an xarray Dataset.
    '''
    primary_hdu = hdulist['primary']
    history_hdu = hdulist['history']
    subint_hdu = hdulist['subint']
    
    data = get_data(subint_hdu, weight)
    data_vars = pol_split(data, subint_hdu.header['pol_type'])
    coords = get_coords(hdulist)
    
    # Add data vars
    duration = subint_hdu.data['tsubint']
    weights = subint_hdu.data['dat_wts']
    weights = weights.reshape(coords['time'].size, coords['freq'].size)
    data_vars['duration'] = (['time'], duration)
    data_vars['weights'] = (['time', 'freq'], weights)
    
    start_time = Time(primary_hdu.header['stt_imjd'], format='mjd')
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
        'channel_offset': subint_hdu.header['nchnoffs'],
        'DM': subint_hdu.header['DM'],
        'RM': subint_hdu.header['RM'],
        'pol_type': subint_hdu.header['pol_type'],
        'start_time': start_time,
        'epoch_type': subint_hdu.header['epochs'],
        'time_var': subint_hdu.header['int_type'],
        'flux_unit': subint_hdu.header['scale'],
        'tbin': history_hdu.data['tbin'][-1],
    }
    
    ds = xr.Dataset(data_vars, coords, attrs)
    
    return ds
    
def get_data(subint_hdu, weight=True):
    '''
    Construct an array of data in meaningful units from the 'SUBINT' HDU
    of a PSRFITS file by applying the scaling, offset, and weights given
    in the file.
    '''
    data = subint_hdu.data['data']
    nsub, npol, nchan, nbin = data.shape
    pol_type = subint_hdu.header['pol_type']
    
    scale = subint_hdu.data['dat_scl']
    offset = subint_hdu.data['dat_offs']
    weights = subint_hdu.data['dat_wts']
    
    # state assumptions explicitly
    assert scale.size == nsub*npol*nchan # Less aggressive check to accomodate templates
    assert offset.size == nsub*npol*nchan
    assert weights.size == nsub*nchan
    
    scale = scale.reshape(nsub, npol, nchan)
    offset = offset.reshape(nsub, npol, nchan)
    weights = weights.reshape(nsub, 1, nchan, 1)
    data = (scale*data.transpose((3,0,1,2)) + offset).transpose((1,2,3,0))
    if weight:
        data = weights*data
    
    return data

def get_coords(hdulist):
    '''
    Get the time, frequency, and phase coordinates from a PSRFITS file.
    '''
    primary_hdu = hdulist['primary']
    history_hdu = hdulist['history']
    subint_hdu = hdulist['subint']
    
    time = subint_hdu.data['offs_sub']
    
    dat_freq = subint_hdu.data['dat_freq']
    freq = np.atleast_1d(dat_freq[0])
    # All other rows should be the same
    if not all(np.all(row == freq) for row in dat_freq):
        msg = 'Not all frequencies match'
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

def native_byteorder(arr):
    '''
    Convert an array to native byte order if it is not already.
    '''
    if arr.dtype.byteorder != '=':
        return arr.byteswap().newbyteorder()
    else:
        return arr

def get_pint_model(ds):
    with tempfile.NamedTemporaryFile('w+') as tp:
        tp.write(ds.source.model)
        tp.flush()
        model = get_model(tp.name)
    return model
