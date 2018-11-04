import numpy as np
import xarray as xr
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

def open(filename, weight=True):
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
    #psrparam_hdu = hdulist['psrparam']
    #polyco_hdu = hdulist['polyco']
    subint_hdu = hdulist['subint']
    
    data = get_data(subint_hdu, weight)
    data_vars = pol_split(data, subint_hdu.header['pol_type'])
    coords = get_coords(hdulist)
    
    # Add data vars
    duration = subint_hdu.data['tsubint']
    weights = subint_hdu.data['dat_wts']
    assert duration.dtype == np.dtype('>f8')
    assert weights.dtype == np.dtype('>f4')
    data_vars['duration'] = (['time'], duration.astype('float64'))
    data_vars['weights'] = (['time', 'freq'], weights.astype('float32'))
    
    assert primary_hdu.header['obsfreq'] == history_hdu.data['ctr_freq'][-1]
    attrs = {'source': primary_hdu.header['src_name'],
             'telescope': primary_hdu.header['telescop'],
             'frontend': primary_hdu.header['frontend'],
             'backend': primary_hdu.header['backend'],
             'center_freq': primary_hdu.header['obsfreq'],
             'DM': subint_hdu.header['DM'],
             'pol_type': subint_hdu.header['pol_type'],
             'start_sec': primary_hdu.header['stt_smjd']}
    
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
    assert scale.shape == (nsub, npol*nchan)
    assert offset.shape == (nsub, npol*nchan)
    assert weights.shape == (nsub, nchan)
    
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
    time = time + primary_hdu.header['stt_offs']
    
    dat_freq = subint_hdu.data['dat_freq']
    assert dat_freq.dtype == np.dtype('>f8')
    freq = dat_freq[0].astype('float64') # convert to native byte order
    # All other rows should be the same
    assert all(np.all(row == freq) for row in dat_freq)
    
    # At least for NANOGrav data, the TBIN in SUBINT is wrong.
    phase = np.arange(history_hdu.data['nbin'][-1])*history_hdu.data['tbin'][-1]
    phase = phase*1000 # s -> ms
    
    coords = {'time': time,
              'freq': freq,
              'phase': phase,
              'MJD': primary_hdu.header['stt_imjd']}
    
    return coords

def pol_split(data, pol_type):
    '''
    Split a PSRFITS dataset into variables representing the polarizations.
    Return the result as a dictionary of data variables suitable for 
    constructing an xarray Dataset.
    '''
    if pol_type == 'AA+BB':
        I, = np.transpose(data, (1, 0, 2, 3))
        data_vars = {'I': (['time', 'freq', 'phase'], AA)}
    elif pol_type == 'AABB':
        AA, BB = np.transpose(data, (1, 0, 2, 3))
        data_vars = {'AA': (['time', 'freq', 'phase'], AA),
                     'BB': (['time', 'freq', 'phase'], BB)}
    elif pol_type == 'AABBCRCI':
        AA, BB, CR, CI = np.transpose(data, (1, 0, 2, 3))
        data_vars = {'AA': (['time', 'freq', 'phase'], AA),
                     'BB': (['time', 'freq', 'phase'], BB),
                     'CR': (['time', 'freq', 'phase'], CR),
                     'CI': (['time', 'freq', 'phase'], CI)}
    elif pol_type == 'IQUV':
        I, Q, U, V = np.transpose(data, (1, 0, 2, 3))
        data_vars = {'I': (['time', 'freq', 'phase'], I),
                     'Q': (['time', 'freq', 'phase'], Q),
                     'U': (['time', 'freq', 'phase'], U),
                     'V': (['time', 'freq', 'phase'], V)}
    else:
        raise ValueError("Polarization type not recognized.")
    
    return data_vars
