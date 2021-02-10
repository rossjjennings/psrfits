import numpy as np

def uniformize(arr, diff):
    unit = gbd(diff)/2
    return unit*np.rint(arr/unit)

def gbd(x):
    mant, expt = np.frexp(x)
    eps = finfo(x).eps
    mant_int = np.int64(mant/eps)
    mant_int = mant_int & ~(mant_int - 1)
    return np.ldexp(mant_int*eps, expt)

def finfo(x):
    try:
        return np.finfo(x.dtype)
    except AttributeError:
        return np.finfo(type(x))
