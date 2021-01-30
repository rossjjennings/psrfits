import numpy as np
import xarray as xr
from pint.models import get_model
import tempfile

def get_pint_model(ds):
    with tempfile.NamedTemporaryFile('w+') as tp:
        tp.write(ds.source.model)
        tp.flush()
        model = get_model(tp.name)
    return model
