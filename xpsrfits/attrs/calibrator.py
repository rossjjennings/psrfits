from textwrap import dedent

class Calibrator:
    __slots__ = 'mode', 'freq', 'duty_cycle', 'phase', 'n_phase'
    
    def __init__(self, mode, freq, duty_cycle, phase, n_phase):
        self.mode = mode
        self.freq = freq
        self.duty_cycle = duty_cycle
        self.phase = phase
        self.n_phase = n_phase
    
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
        return f'{self.mode} [...]'.strip()
    
    def __repr__(self):
        description = f'''
        <xpsrfits.Calibrator>
            mode:       {self.mode}
            freq:       {self.freq}
            duty_cycle: {self.duty_cycle}
            phase:      {self.phase}
            n_phase:    {self.n_phase}
        '''
        return dedent(description).strip('\n')
