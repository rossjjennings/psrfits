from textwrap import indent

from .attrcollection import AttrCollection
from .polyco import PolycoHistory
from .t2predict import ChebyModelSet

class Source(AttrCollection):
    __slots__ = 'name', 'model', 'polyco_history', 'predictor'
    
    @classmethod
    def from_hdulist(cls, hdulist):
        polyco_history = None
        predictor = None
        if 'polyco' in hdulist:
            polyco_history = PolycoHistory(hdulist['polyco'].data)
        if 't2predict' in hdulist:
            predictor = ChebyModelSet.parse([line[0] for line in hdulist['t2predict'].data])
        return cls(
            name = hdulist['primary'].header['src_name'],
            model = '\n'.join(line[0] for line in hdulist['psrparam'].data),
            polyco_history = polyco_history,
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
