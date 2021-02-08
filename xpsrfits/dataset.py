import xarray as xr

class Dataset:
    __slots__ = ('_ds')
    
    def __init__(self, *args, **kwargs):
        self._ds = xr.Dataset(*args, **kwargs)
    
    def __repr__(self):
        lines = repr(self._ds).split('\n')
        lines[0] = '<xpsrfits.Dataset>'
        return '\n'.join(lines)
    
    def _repr_html_(self):
        html = self._ds._repr_html_()
        return 'xpsrfits.Dataset'.join(html.split('xarray.Dataset'))
    
    def __getattr__(self, name):
        forwarded = (
            name in ['data_vars', 'coords', 'attrs']
            or name in self._ds.data_vars
            or name in self._ds.coords
            or name in self._ds.attrs
        )
        if forwarded:
            return getattr(self._ds, name)
        raise AttributeError(f"'Dataset' object has no attribute '{name}'")
