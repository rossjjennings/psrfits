from textwrap import indent

from .attrcollection import AttrCollection, maybe_missing

class Beam(AttrCollection):
    __slots__ = 'beam_id', 'center_id', 'major_axis', 'minor_axis', 'pos_angle'
    
    @classmethod
    def from_header(cls, header):
        return cls(
            beam_id = maybe_missing(header['ibeam']), # ''
            center_id = maybe_missing(header['pnt_id']), # ''
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
    
    def header_cards(self):
        return {
            'ibeam': self.beam_id,
            'bmaj': self.major_axis,
            'bmin': self.minor_axis,
            'bpa': self.pos_angle,
            'pnt_id': self.center_id,
        }
