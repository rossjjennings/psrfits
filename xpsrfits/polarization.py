import numpy as np
import xarray as xr
from xpsrfits.dataset import Dataset

def pol_split(data, pol_type):
    '''
    Split a PSRFITS dataset into variables representing the polarizations.
    Return the result as a dictionary of data variables suitable for 
    constructing an xarray Dataset.
    '''
    if pol_type in ['AA+BB', 'INTEN']:
        I, = np.swapaxes(data, 0, 1)
        data_vars = {'I': (['time', 'freq', 'phase'], I)}
    elif pol_type == 'AABB':
        AA, BB = np.swapaxes(data, 0, 1)
        data_vars = {'AA': (['time', 'freq', 'phase'], AA),
                     'BB': (['time', 'freq', 'phase'], BB)}
    elif pol_type == 'AABBCRCI':
        AA, BB, CR, CI = np.swapaxes(data, 0, 1)
        data_vars = {'AA': (['time', 'freq', 'phase'], AA),
                     'BB': (['time', 'freq', 'phase'], BB),
                     'CR': (['time', 'freq', 'phase'], CR),
                     'CI': (['time', 'freq', 'phase'], CI)}
    elif pol_type == 'IQUV':
        I, Q, U, V = np.swapaxes(data, 0, 1)
        data_vars = {'I': (['time', 'freq', 'phase'], I),
                     'Q': (['time', 'freq', 'phase'], Q),
                     'U': (['time', 'freq', 'phase'], U),
                     'V': (['time', 'freq', 'phase'], V)}
    else:
        raise ValueError("Polarization type '{}' not recognized.".format(pol_type))
    
    return data_vars

def get_pols(ds):
    '''
    Return a list the polarizations present in the dataset `ds`.
    '''
    if ds.pol_type in ['AA+BB', 'INTEN']:
        return ['I']
    elif ds.pol_type == 'AABB':
        return ['AA', 'BB']
    elif ds.pol_type == 'AABBCRCI':
        return ['AA', 'BB', 'CR', 'CI']
    elif ds.pol_type == 'IQUV':
        return ['I', 'Q', 'U', 'V']
    else:
        raise ValueError("Polarization type '{}' not recognized.".format(ds.pol_type))

def pscrunch(ds):
    '''
    Return a dataset containing only the total intensity component of the input.
    '''
    new_data_vars = dict(ds.data_vars)
    for pol in get_pols(ds):
        del new_data_vars[pol]
    if ds.pol_type in ['AA+BB', 'INTEN', 'IQUV']:
        new_data_vars['I'] = ds.I
    elif ds.pol_type in ['AABB', 'AABBCRCI']:
        new_data_vars['I'] = ds.AA + ds.BB
    else:
        raise ValueError("Polarization type '{}' not recognized.".format(ds.pol_type))
    new_attrs = ds.attrs.copy()
    new_attrs['pol_type'] = 'AA+BB'
    return Dataset(new_data_vars, ds.coords, new_attrs)

def to_stokes(ds):
    '''
    Transform coherence (AABBCRCI) data to Stokes parameters (IQUV).
    If input is already Stokes, leave it alone.
    If input has one or two polarizations, return I only.
    '''
    if ds.pol_type == 'IQUV':
        return ds
    elif ds.pol_type in ['AABB', 'AA+BB', 'INTEN']:
        return pscrunch(ds)
    elif ds.pol_type != 'AABBCRCI':
        raise ValueError("Polarization type '{}' not recognized.".format(ds.pol_type))
    new_data_vars = dict(ds.data_vars)
    for pol in get_pols(ds):
        del new_data_vars[pol]
    if ds.frontend.feed_poln == 'LIN':
        new_data_vars['I'] = ds.AA + ds.BB
        new_data_vars['Q'] =  ds.AA - ds.BB
        new_data_vars['U'] = 2*ds.CR
        new_data_vars['V'] = 2*ds.CI
    elif ds.frontend.feed_poln == 'CIRC':
        new_data_vars['I'] = ds.AA + ds.BB
        new_data_vars['Q'] = 2*ds.CR
        new_data_vars['U'] = 2*ds.CI
        new_data_vars['V'] = ds.AA - ds.BB
    new_attrs = ds.attrs.copy()
    new_attrs['pol_type'] = 'IQUV'
    return Dataset(new_data_vars, ds.coords, new_attrs)
