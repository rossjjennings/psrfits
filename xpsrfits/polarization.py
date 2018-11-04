import numpy as np
import xarray as xr

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
