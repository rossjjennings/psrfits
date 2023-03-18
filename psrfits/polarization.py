import numpy as np
#from psrfits.dataset import Dataset

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
        new_data_vars['I'] = (['time', 'freq', 'phase'], ds.I)
    elif ds.pol_type in ['AABB', 'AABBCRCI']:
        new_data_vars['I'] = (['time', 'freq', 'phase'], ds.AA + ds.BB)
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
        new_data_vars['I'] = (['time', 'freq', 'phase'], ds.AA + ds.BB)
        new_data_vars['Q'] =  (['time', 'freq', 'phase'], ds.AA - ds.BB)
        new_data_vars['U'] = (['time', 'freq', 'phase'], 2*ds.CR)
        new_data_vars['V'] = (['time', 'freq', 'phase'], 2*ds.CI)
    elif ds.frontend.feed_poln == 'CIRC':
        new_data_vars['I'] = (['time', 'freq', 'phase'], ds.AA + ds.BB)
        new_data_vars['Q'] = (['time', 'freq', 'phase'], 2*ds.CR)
        new_data_vars['U'] = (['time', 'freq', 'phase'], 2*ds.CI)
        new_data_vars['V'] = (['time', 'freq', 'phase'], ds.AA - ds.BB)
    new_attrs = ds.attrs.copy()
    new_attrs['pol_type'] = 'IQUV'
    return Dataset(new_data_vars, ds.coords, new_attrs)
