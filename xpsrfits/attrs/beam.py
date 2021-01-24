from textwrap import indent

class Beam:
    __slots__ = 'beam_id', 'center_id', 'major_axis', 'minor_axis', 'pos_angle'
    
    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs[name])
    
    @classmethod
    def from_header(cls, header):
        return cls(
            beam_id = header['ibeam'],
            center_id = header['pnt_id'],
            major_axis = header['bmaj'],
            minor_axis = header['bmin'],
            pos_angle = header['bpa'],
        )
    
    def _repr_items(self):
        max_len = max(len(name) for name in self.__slots__)
        description = ""
        for name in self.__slots__:
            key = f"{name}:"
            description += f"{key:<{max_len + 2}}{getattr(self, name)}\n"
        return description
    
    def __str__(self):
        if self.beam_id is None or self.beam_id == '':
            return '<Beam>'
        else:
            return f'<Beam {self.beam_id}>'
    
    def __repr__(self):
        description = "<xpsrfits.Beam>\n"
        description += indent(self._repr_items(), '    ')
        return description
