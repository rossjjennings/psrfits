from textwrap import indent

class Backend:
    __slots__ = 'name', 'config', 'phase', 'dcc', 'delay', 'cycle_time'
    
    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs[name])
    
    @classmethod
    def from_header(cls, header):
        return cls(
            name = header['backend'],
            config = header['beconfig'],
            phase = header['be_phase'],
            dcc = header['be_dcc'],
            delay = header['be_delay'],
            cycle_time = header['tcycle'],
        )
    
    def _repr_items(self):
        max_len = max(len(name) for name in self.__slots__)
        description = ""
        for name in self.__slots__:
            key = f"{name}:"
            description += f"{key:<{max_len + 2}}{getattr(self, name)}\n"
        return description
    
    def __str__(self):
        return f'<{self.name}>'
    
    def __repr__(self):
        description = "<xpsrfits.Backend>\n"
        description += indent(self._repr_items(), '    ')
        return description
