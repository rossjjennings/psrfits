from textwrap import indent

from .attrcollection import AttrCollection
from .polyco import Polyco

class Source(AttrCollection):
    __slots__ = 'name', 'model', 'polyco', 'predictor'
    
    @classmethod
    def from_hdulist(cls, hdulist):
        polyco = None
        predictor = None
        if 'polyco' in hdulist:
            polyco = Polyco(hdulist['polyco'].data)
        if 't2predict' in hdulist:
            predictor = '\n'.join(line[0] for line in hdulist['t2predict'].data)
        return cls(
            name = hdulist['primary'].header['src_name'],
            model = '\n'.join(line[0] for line in hdulist['psrparam'].data),
            polyco = polyco,
            predictor = predictor,
        )
    
    def __str__(self):
        return f'<{self.name}>'
    
    def __repr__(self):
        description = "<psrfits.Source>\n"
        description += indent(self._repr_items(), '    ')
        return description
    
    def header_cards(self):
        return {
            'src_name': self.name,
        }
