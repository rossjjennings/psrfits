from textwrap import indent

from .attrcollection import AttrCollection

class Beam(AttrCollection):
    __slots__ = 'beam_id', 'center_id', 'major_axis', 'minor_axis', 'pos_angle'
    
    @classmethod
    def from_header(cls, header):
        return cls(
            beam_id = header['ibeam'],
            center_id = header['pnt_id'],
            major_axis = header['bmaj'],
            minor_axis = header['bmin'],
            pos_angle = header['bpa'],
        )
    
    def __str__(self):
        if self.beam_id is None or self.beam_id == '':
            return '<Beam>'
        else:
            return f'<Beam {self.beam_id}>'
    
    def __repr__(self):
        description = "<xpsrfits.Beam>\n"
        description += indent(self._repr_items(), '    ')
        return description
