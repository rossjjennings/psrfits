from textwrap import indent

class Frontend:
    __slots__ = 'name', 'n_pol', 'feed_poln', 'handedness', 's_angle', 'xy_phase'
    
    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs[name])
    
    @classmethod
    def from_header(cls, header):
        return cls(
            name = header['frontend'],
            n_pol = header['nrcvr'],
            feed_poln = header['fd_poln'],
            handedness = header['fd_hand'],
            s_angle = header['fd_sang'],
            xy_phase = header['fd_xyph'],
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
        description = "<xpsrfits.Frontend>\n"
        description += indent(self._repr_items(), '    ')
        return description
