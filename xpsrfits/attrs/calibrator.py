from textwrap import indent

from .attrcollection import AttrCollection, maybe_missing, if_missing

class Calibrator(AttrCollection):
    __slots__ = 'mode', 'freq', 'duty_cycle', 'phase', 'n_phase'
    
    @classmethod
    def from_header(cls, header):
        return cls(
            mode = maybe_missing(header['cal_mode']),
            freq = maybe_missing(header['cal_freq']),
            duty_cycle = maybe_missing(header['cal_dcyc']),
            phase = maybe_missing(header['cal_phs']),
            n_phase = maybe_missing(header['cal_nphs']),
        )
    
    def __str__(self):
        if self.mode is None or self.mode == '':
            return '<Calibrator>'
        else:
            return f'<{self.mode} mode calibrator>'
    
    def __repr__(self):
        description = "<xpsrfits.Calibrator>\n"
        description += indent(self._repr_items(), '    ')
        return description
    
    def header_cards(self):
        return {
            'cal_mode': if_missing('', self.mode),
            'cal_freq': if_missing('*', self.freq),
            'cal_dcyc': if_missing('*', self.duty_cycle),
            'cal_phs': if_missing('*', self.phase),
            'cal_nphs': if_missing('*', self.n_phase),
        }
