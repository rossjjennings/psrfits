import numpy as np
import xarray as xr

def pol_split(data, pol_type):
    '''
    Split a PSRFITS dataset into variables representing the polarizations.
    Return the result as a dictionary of data variables suitable for 
    constructing an xarray Dataset.
    '''
    if pol_type == 'AA+BB':
        I, = np.transpose(data, (1, 0, 2, 3))
        data_vars = {'I': (['time', 'freq', 'phase'], I)}
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

def get_pols(ds):
    '''
    Return a list the polarizations present in the dataset `ds`.
    '''
    if ds.pol_type == 'AA+BB':
        return ['I']
    elif ds.pol_type == 'AABB':
        return ['AA', 'BB']
    elif ds.pol_type == 'AABBCRCI':
        return ['AA', 'BB', 'CR', 'CI']
    elif ds.pol_type == 'IQUV':
        return ['I', 'Q', 'U', 'V']
    else:
        raise ValueError("Polarization type not recognized.")

def pscrunch(ds):
    '''
    Return a dataset containing only the total intensity component of the input.
    '''
    if ds.pol_type in ['AA+BB', 'IQUV']:
        new_data_vars = {'I': ds.I}
    elif ds.pol_type == ['AABB', 'AABBCRCI']:
        new_data_vars = {'I': ds.AA + ds.BB}
    else:
        raise ValueError("Polarization type not recognized.")
    new_attrs = ds.attrs.copy()
    new_attrs['pol_type'] = 'AA+BB'
    return xr.Dataset(new_data_vars, ds.coords, new_attrs)

def coherence_to_stokes(ds):
    '''
    Transform coherence (AABBCRCI) data to Stokes parameters (IQUV).
    '''
    if ds.pol_type != 'AABBCRCI':
        raise ValueError("Input is not coherence data!")
    if ds.fd_poln == 'LIN':
        new_data_vars = {'I': ds.AA + ds.BB,
                         'Q': ds.AA - ds.BB,
                         'U': 2*ds.CR,
                         'V': 2*ds.CI      }
    elif ds.fd_poln == 'CIRC':
        new_data_vars = {'I': ds.AA + ds.BB,
                         'Q': 2*ds.CR,
                         'U': 2*ds.CI,
                         'V': ds.AA - ds.BB}
    new_attrs = ds.attrs.copy()
    new_attrs['pol_type'] = 'IQUV'
    return xr.Dataset(new_data_vars, ds.coords, new_attrs)
