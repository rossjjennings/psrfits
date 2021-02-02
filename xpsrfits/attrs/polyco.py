import numpy as np
from textwrap import indent
from datetime import datetime
from astropy.time import Time

from .attrcollection import AttrCollection, maybe_missing, if_missing

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
    
    def as_table(self):
        return np.array(
            [(
                entry.date_as_string(),
                if_missing('', entry.version),
                entry.span,
                entry.n_coeffs,
                entry.n_blocks,
                entry.obscode,
                entry.ref_freq,
                entry.start_phase,
                entry.ref_mjd.mjd,
                entry.ref_phase,
                entry.ref_f0,
                entry.log10_fit_err,
                entry.coeffs,
            ) for entry in self],
            dtype = (np.record, [
                ('DATE_PRO', 'S24'),
                ('POLYVER', 'S16'),
                ('NSPAN', '>i2'),
                ('NCOEF', '>i2'),
                ('NPBLK', '>i2'),
                ('NSITE', 'S8'),
                ('REF_FREQ', '>f8'),
                ('PRED_PHS', '>f8'),
                ('REF_MJD', '>f8'),
                ('REF_PHS', '>f8'),
                ('REF_F0', '>f8'),
                ('LGFITERR', '>f8'),
                ('COEFF', '>f8', (15,)),
            ]),
        )

class PolycoEntry(AttrCollection):
    __slots__ = (
        'date',
        'version',
        'span',
        'n_coeffs',
        'n_blocks',
        'obscode',
        'ref_freq',
        'start_phase',
        'ref_mjd',
        'ref_phase',
        'ref_f0',
        'log10_fit_err',
        'coeffs',
    )
    
    @classmethod
    def from_table(cls, table):
        entries = [{} for i in range(table.size)]
        for i in range(table.size):
            date = maybe_missing(table['date_pro'][i])
            if date is not None:
                date = Time(date)
            entries[i] = {
                'date': date,
                'version': maybe_missing(table['polyver'][i]),
                'span': table['nspan'][i],
                'n_coeffs': table['ncoef'][i],
                'n_blocks': table['npblk'][i],
                'obscode': table['nsite'][i],
                'ref_freq': table['ref_freq'][i],
                'start_phase': table['pred_phs'][i],
                'ref_mjd': Time(table['ref_mjd'][i], format='mjd'),
                'ref_phase': table['ref_phs'][i],
                'ref_f0': table['ref_f0'][i],
                'log10_fit_err': table['lgfiterr'][i],
                'coeffs': table['coeff'][i],
            }
        return [cls(**entry) for entry in entries]
    
    def __str__(self):
        return f'<PolycoEntry>'
    
    def __repr__(self):
        description = "<xpsrfits.PolycoEntry>\n"
        description += indent(self._repr_items(), '    ')
        return description
    
    def date_as_string(self):
        if self.date is None:
            return ''
        else:
            return datetime.strftime(self.date.datetime, '%a %b %d %H:%M:%S %Y')
