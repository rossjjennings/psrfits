import numpy as np

def get_pols(ds):
    '''
    Return a list of the polarizations present in the dataset `ds`.
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
        raise ValueError(f"Polarization type '{ds.pol_type}' not recognized")

def pscrunch(ds, inplace=False):
    '''
    Return a dataset containing only the total intensity component of the input.
    Performs a copy unless `inplace=True`.
    '''
    new_ds = ds if inplace else ds.copy()

    if ds.pol_type in ['AA+BB', 'INTEN', 'IQUV']:
        return new_ds
    elif ds.pol_type in ['AABB', 'AABBCRCI']:
        new_ds.I = ds.AA + ds.BB
        new_ds.data = np.stack([new_ds.I])
        for pol in get_pols(ds):
            delattr(new_ds, pol)
        new_ds.pol_type='AA+BB'
        return new_ds
    else:
        raise ValueError(f"Polarization type '{ds.pol_type}' not recognized")

def to_stokes(ds, inplace=False):
    '''
    Transform coherence (AABBCRCI) data to Stokes parameters (IQUV).
    If input is already Stokes, leave it alone.
    If input has one or two polarizations, return I only.
    Performs a copy unless `inplace=True`.
    '''
    new_ds = ds if inplace else ds.copy()

    if ds.pol_type in ['AABB', 'AA+BB', 'INTEN', 'IQUV']:
        return new_ds
    elif ds.pol_type == 'AABBCRCI':
        if ds.frontend.feed_poln == 'LIN':
            new_ds.I = ds.AA + ds.BB
            new_ds.Q = ds.AA - ds.BB
            new_ds.U = 2*ds.CR
            new_ds.V = 2*ds.CI
        elif ds.frontend.feed_poln == 'CIRC':
            new_ds.I = ds.AA + ds.BB
            new_ds.Q = 2*ds.CR
            new_ds.U = 2*ds.CI
            new_ds.V = ds.AA - ds.BB
        else:
            raise ValueError(f"Feed polarization '{ds.frontend.feed_poln}' not recognized")
        new_ds.data = np.stack([new_ds.I, new_ds.Q, new_ds.U, new_ds.V])
        for pol in get_pols(ds):
            delattr(new_ds, pol)
        new_ds.pol_type = 'IQUV'
        return new_ds
    else:
        raise ValueError(f"Polarization type '{ds.pol_type}' not recognized")
