from psrfits.formatting import fmt_items, fmt_array
from textwrap import indent

class Dataset:
    __slots__ = ('data_vars', 'coords', 'attrs')
    
    def __init__(self, data_vars, coords, attrs):
        self.data_vars = data_vars
        self.coords = coords
        self.attrs = attrs
    
    def __repr__(self):
        return ("<psrfits.Dataset>\n"
            f"Dimensions: ({', '.join(f'{key}: {len(value)}' for key, value in self.coords.items())})\n"
            f"Coordinates:\n{indent(fmt_items({name: fmt_coord(name, coord) for name, coord in self.coords.items()}), '    ')}"
            f"Data variables:\n{indent(fmt_items({key: fmt_datavar(value) for key, value in self.data_vars.items()}), '    ')}"
            f"Attributes:\n{indent(fmt_items(self.attrs), '    ')}"
        )
    
    def __getattr__(self, name):
        try:
            return self.data_vars[name][1]
        except KeyError:
            try:
                return self.coords[name]
            except KeyError:
                try:
                    return self.attrs[name]
                except KeyError:
                    raise AttributeError(f"'Dataset' object has no attribute '{name}'")

def fmt_coord(name, coord):
    dims_str = f"({name})"
    dims_dtype = dims_str + ' ' + str(coord.dtype) + ' '
    return dims_dtype + fmt_array(coord, 65 - len(dims_dtype))
    

def fmt_datavar(datavar):
    dims, var = datavar
    dims_str = f"({', '.join(dims)})"
    dims_dtype = dims_str + ' ' + str(var.dtype) + ' '
    return dims_dtype + fmt_array(var, 65 - len(dims_dtype))
