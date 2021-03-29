from xpsrfits.formatting import fmt_items

class AttrCollection:
    __slots__ = tuple()
    
    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs[name])
    
    def _repr_items(self):
        mapping = {name: getattr(self, name) for name in self.__slots__}
        return fmt_items(mapping)

def maybe_missing(item):
    if item in ['', '*', 'N/A', 'UNSET', 'UNSETTUNSET', 'UNKNOWN', 'NONE']:
        return None
    else:
        return item

def if_missing(alt_text, item):
    if item is None:
        return alt_text
    else:
        return item
