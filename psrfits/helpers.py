import numpy as np
from pint.models import get_model
import pywt
import tempfile

def get_pint_model(ds):
    with tempfile.NamedTemporaryFile('w+') as tp:
        tp.write(ds.source.model)
        tp.flush()
        model = get_model(tp.name)
    return model

# Modified from code written by E. Fonseca for PulsePortraiture
def wavelet_smooth(prof, wavelet='db8', nlevel=5, threshtype='hard', fact=1.0):
    nbin = prof.shape[-1]
    # Translation-invariant (stationary) wavelet transform/denoising
    coeffs = np.array(pywt.swt(prof, wavelet, level=nlevel, start_level=0, axis=-1))
    # Get threshold value
    lopt = fact * (np.median(np.abs(coeffs[0])) / 0.6745) * np.sqrt(2 * np.log(nbin))
    # Do wavelet thresholding
    coeffs = pywt.threshold(coeffs, lopt, mode=threshtype, substitute=0.0)
    # Reconstruct data
    smooth_prof = pywt.iswt(list(map(tuple, coeffs)), wavelet)
    return smooth_prof
