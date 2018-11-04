import numpy as np
import xarray as xr
from xpsrfits.polarization import get_pols

def remove_baseline(ds, method='median'):
    '''
    Remove the frequency-dependent baseline from an observation.
    '''
    new_data_vars = {}
    for pol in get_pols(ds):
        arr = ds.data_vars[pol]
        baseline = arr.median(dim='phase')
        adjusted_arr = arr - baseline
        new_data_vars.update({pol: adjusted_arr})
    
    return xr.Dataset(new_data_vars, ds.coords, ds.attrs)
