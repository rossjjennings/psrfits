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
    if len(lines) > 1 and not isinstance(item, np.ndarray):
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
