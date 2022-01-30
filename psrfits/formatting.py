import numpy as np
from astropy.coordinates import SkyCoord
from textwrap import indent

def fmt_items(mapping):
    max_len = max((len(name) for name in mapping if mapping[name] is not None), default=0)
    description = ""
    for name, item in mapping.items():
        if item is not None:
            key = f"{name}:"
            description += f"{key:<{max_len + 3}}"
            lines = fmt_inline(item).split('\n')
            first_line = lines[0] + '\n'
            rest = '\n'.join(lines[1:])
            rest = indent(rest, ' '*(max_len + 3))
            description += first_line + (rest if lines else '')
    return description

def fmt_inline(item):
    if isinstance(item, SkyCoord):
        item_str = fmt_skycoord(item)
    else:
        item_str = str(item)
    lines = item_str.split('\n')
    if isinstance(item, np.ndarray) and len(item.shape) > 0:
        dims = str(item.shape)
        dims_dtype = dims + ' ' + str(item.dtype) + ' '
        return dims_dtype + fmt_array(item, 65 - len(dims_dtype))
    elif len(lines) > 1:
        return f'{lines[0]} [...]'
    else:
        return item_str

def fmt_skycoord(skycoord):
    class_name = skycoord.__class__.__name__
    frame_name = skycoord.frame.__class__.__name__
    component_names = list(skycoord.representation_component_names)
    if 'distance' in component_names and skycoord.distance == 1.0:
        component_names.remove('distance')
    values = ', '.join(f'{name}={getattr(skycoord, name):g}' for name in component_names)
    return f'<{class_name} ({frame_name}): {values}>'

def fmt_element(elt):
    if hasattr(elt, "dtype") and np.issubdtype(elt.dtype, np.floating):
        return f"{elt.item():.4}"
    else:
        return str(elt)

def fmt_array(array, max_width):
    front = []
    back = []
    cols_remaining = max_width
    front_idx = 0
    back_idx = array.size - 1
    while True:
        # add something to the front
        elt = array.flat[front_idx]
        elt_str = fmt_element(elt)
        cols_remaining -= (len(elt_str) + 1)
        front_idx += 1
        
        if front_idx > back_idx:
            return ' '.join(front) + ' ' + ' '.join(back[::-1])
        elif cols_remaining < 3:
            return ' '.join(front) + ' ... ' + ' '.join(back[::-1])
        else:
            front.append(elt_str)
        
        # add something to the back
        elt = array.flat[back_idx]
        elt_str = fmt_element(elt)
        cols_remaining -= (len(elt_str) + 1)
        back_idx -= 1
        
        if back_idx < front_idx:
            return ' '.join(front) + ' ' + ' '.join(back[::-1])
        elif cols_remaining < 3:
            return ' '.join(front) + ' ... ' + ' '.join(back[::-1])
        else:
            back.append(elt_str)
