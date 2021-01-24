from astropy.coordinates import SkyCoord

class AttrCollection:
    __slots__ = tuple()
    
    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs[name])
    
    def _repr_items(self):
        max_len = max(len(name) for name in self.__slots__)
        description = ""
        for name in self.__slots__:
            item = getattr(self, name)
            if item is not None:
                key = f"{name}:"
                description += f"{key:<{max_len + 2}}{self._fmt_item(item)}\n"
        return description
    
    def _fmt_item(self, item):
        if isinstance(item, SkyCoord):
            item_str = fmt_skycoord(item)
        else:
            item_str = str(item)
        lines = item_str.split('\n')
        if len(lines) > 1:
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

def maybe_missing(item):
    if item in ['', '*', 'N/A', 'UNSET', 'UNKNOWN', 'NONE']:
        return None
    else:
        return item
