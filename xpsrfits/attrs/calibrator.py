from textwrap import indent

from .attrcollection import AttrCollection

class Calibrator(AttrCollection):
    __slots__ = 'mode', 'freq', 'duty_cycle', 'phase', 'n_phase'
    
    @classmethod
    def from_header(cls, header):
        return cls(
            mode = header['cal_mode'],
            freq = header['cal_freq'],
            duty_cycle = header['cal_dcyc'],
            phase = header['cal_phs'],
            n_phase = header['cal_nphs'],
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
