from textwrap import indent

from .attrcollection import AttrCollection, maybe_missing, if_missing

class Backend(AttrCollection):
    __slots__ = 'name', 'config', 'phase', 'dcc', 'delay', 'cycle_time'
    
    @classmethod
    def from_header(cls, header):
        return cls(
            name = header['backend'],
            config = maybe_missing(header['beconfig']),
            phase = header['be_phase'],
            dcc = bool(header['be_dcc']),
            delay = header['be_delay'],
            cycle_time = header['tcycle'],
        )
    
    def __str__(self):
        return f'<{self.name}>'
    
    def __repr__(self):
        description = "<psrfits.Backend>\n"
        description += indent(self._repr_items(), '    ')
        return description
    
    def header_cards(self):
        return {
            'backend': self.name,
            'beconfig': if_missing('N/A', self.config),
            'be_phase': self.phase,
            'be_dcc': int(self.dcc),
            'be_delay': self.delay,
            'tcycle': self.cycle_time,
        }
