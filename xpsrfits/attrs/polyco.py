from textwrap import indent
from datetime import datetime
from astropy.time import Time

from .attrcollection import AttrCollection, maybe_missing

class Polyco:
    def __init__(self, entries):
        self.entries = PolycoEntry.from_table(entries)
    
    @classmethod
    def from_hdu(cls, hdu):
        return cls(hdu.data)
    
    def __str__(self):
        return '<Polyco>'
    
    def __repr__(self):
        description = "<xpsrfits.Polyco>\nLatest entry:\n"
        description += indent(self.entries[-1]._repr_items(), '    ')
        return description
    
    def __getattr__(self, name):
        return getattr(self.entries[-1], name)
    
    def __getitem__(self, key):
        return self.entries[key]
    
    def __len__(self):
        return len(self.entries)
    
    def __iter__(self):
        for entry in self.entries:
            yield entry

class PolycoEntry(AttrCollection):
    __slots__ = (
        'date',
        'version',
        'span',
        'obscode',
        'ref_freq',
        'start_phase',
        'ref_mjd',
        'ref_phase',
        'ref_f0',
        'coeffs',
    )
    
    @classmethod
    def from_table(cls, table):
        entries = [{} for i in range(table.size)]
        for i in range(table.size):
            date = maybe_missing(table['date_pro'][i]) # ''
            if date is not None:
                date = Time(date)
            entries[i] = {
                'date': date,
                'version': maybe_missing(table['polyver'][i]), # ''
                'span': table['nspan'][i],
                'obscode': table['nsite'][i],
                'ref_freq': table['ref_freq'][i],
                'start_phase': table['pred_phs'][i],
                'ref_mjd': Time(table['ref_mjd'][i], format='mjd'),
                'ref_phase': table['ref_phs'][i],
                'ref_f0': table['ref_f0'][i],
                'coeffs': table['coeff'][i],
            }
            return [cls(**entry) for entry in entries]
        return entries
    
    def __str__(self):
        return f'<PolycoEntry>'
    
    def __repr__(self):
        description = "<xpsrfits.PolycoEntry>\n"
        description += indent(self._repr_items(), '    ')
        return description
