from textwrap import indent

from .attrcollection import AttrCollection

class Frontend(AttrCollection):
    __slots__ = 'name', 'n_pol', 'feed_poln', 'handedness', 's_angle', 'xy_phase'
    
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
    
    def __str__(self):
        return f'<{self.name}>'
    
    def __repr__(self):
        description = "<xpsrfits.Frontend>\n"
        description += indent(self._repr_items(), '    ')
        return description
