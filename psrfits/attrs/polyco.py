import numpy as np
from numpy.polynomial import polynomial
from textwrap import indent
from datetime import datetime
from astropy.time import Time

from .attrcollection import AttrCollection, maybe_missing, if_missing

class PolycoHistory:
    def __init__(self, records):
        self.entries = []
        for rec in records:
            self.entries.append(PolycoModel.from_record(rec))
    
    @classmethod
    def from_hdu(cls, hdu):
        return cls(hdu.data)
    
    def __str__(self):
        return '<PolycoHistory>'
    
    def __repr__(self):
        description = "<psrfits.PolycoHistory>\nLatest entry:\n"
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
                entry.site,
                entry.ref_freq,
                entry.start_phase,
                entry.ref_mjd,
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

class PolycoModel(AttrCollection):
    __slots__ = (
        'date_produced',
        'version',
        'span',
        'n_coeffs',
        'n_blocks',
        'site',
        'ref_freq',
        'start_phase',
        'ref_mjd',
        'ref_phase',
        'ref_f0',
        'log10_fit_err',
        'coeffs',
    )
    
    @classmethod
    def from_record(cls, rec):
        date = maybe_missing(rec['DATE_PRO'])
        if date is not None:
            date = Time(date)
        return cls(
            date_produced = date,
            version = rec['POLYVER'],
            span = rec['NSPAN'],
            n_coeffs = rec['NCOEF'],
            n_blocks = rec['NPBLK'],
            site = rec['NSITE'],
            ref_freq = rec['REF_FREQ'],
            start_phase = rec['PRED_PHS'],
            ref_mjd = rec['REF_MJD'],
            ref_phase = rec['REF_PHS'],
            ref_f0 = rec['REF_F0'],
            log10_fit_err = rec['LGFITERR'],
            coeffs = rec['COEFF'],
        )
    
    def __str__(self):
        return f'<PolycoModel>'
    
    def __repr__(self):
        description = "<psrfits.PolycoModel>\n"
        description += indent(self._repr_items(), '    ')
        return description

    def __call__(self, mjd, check_bounds=True):
        dt = self.dt(mjd, check_bounds)
        phase = self.ref_phase + dt*60*self.ref_f0 + polynomial.polyval(dt, self.coeffs)
        return phase

    def f0(self, mjd, check_bounds=True):
        dt = self.dt(mjd, check_bounds)

        der_coeffs = polynomial.polyder(self.coeffs)
        f0 = self.ref_f0 + polynomial.polyval(dt, der_coeffs)/60
        return f0
    
    def dt(self, mjd, check_bounds=True):
        mjd_start = self.ref_mjd - self.span/1440
        mjd_end = self.ref_mjd + self.span/1440
        if check_bounds and np.any((mjd < mjd_start) | (mjd > mjd_end)):
            raise ValueError(f'MJD {mjd[(mjd < mjd_start) | (mjd > mjd_end)]} out of bounds.')

        dt = (mjd - self.ref_mjd)*1440 # minutes
        return dt

    def date_as_string(self):
        if self.date_produced is None:
            return ''
        else:
            return datetime.strftime(self.date_produced.datetime, '%a %b %d %H:%M:%S %Y')
