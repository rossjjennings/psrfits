from textwrap import indent

class Calibrator:
    __slots__ = 'mode', 'freq', 'duty_cycle', 'phase', 'n_phase'
    
    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs[name])
    
    @classmethod
    def from_header(cls, header):
        return cls(
            mode = header['cal_mode'],
            freq = header['cal_freq'],
            duty_cycle = header['cal_dcyc'],
            phase = header['cal_phs'],
            n_phase = header['cal_nphs'],
        )
    
    def _repr_items(self):
        max_len = max(len(name) for name in self.__slots__)
        description = ""
        for name in self.__slots__:
            key = f"{name}:"
            description += f"{key:<{max_len + 2}}{getattr(self, name)}\n"
        return description
    
    def __str__(self):
        if self.mode is None or self.mode == '':
            return '<Calibrator>'
        else:
            return f'<{self.mode} mode calibrator>'
    
    def __repr__(self):
        description = "<xpsrfits.Calibrator>\n"
        description += indent(self._repr_items(), '    ')
        return description

