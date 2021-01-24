from textwrap import indent

from .attrcollection import AttrCollection, maybe_missing

class Backend(AttrCollection):
    __slots__ = 'name', 'config', 'phase', 'dcc', 'delay', 'cycle_time'
    
    @classmethod
    def from_header(cls, header):
        return cls(
            name = header['backend'],
            config = maybe_missing(header['beconfig']), # N/A
            phase = header['be_phase'],
            dcc = header['be_dcc'],
            delay = header['be_delay'],
            cycle_time = header['tcycle'],
        )
    
    def __str__(self):
        return f'<{self.name}>'
    
    def __repr__(self):
        description = "<xpsrfits.Backend>\n"
        description += indent(self._repr_items(), '    ')
        return description
