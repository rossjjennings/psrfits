from textwrap import dedent

class Backend:
    __slots__ = 'name', 'config', 'phase', 'dcc', 'delay', 'cycle_time'
    
    def __init__(self, name, config, phase, dcc, delay, cycle_time):
        self.name = name
        self.config = config
        self.phase = phase
        self.dcc = dcc
        self.delay = delay
        self.cycle_time = cycle_time
    
    @classmethod
    def from_header(cls, header):
        return cls(
            name = header['backend'],
            config = header['beconfig'],
            phase = header['be_phase'],
            dcc = header['be_dcc'],
            delay = header['be_delay'],
            cycle_time = header['tcycle'],
        )
    
    def __str__(self):
        return f'{self.name} [...]'
    
    def __repr__(self):
        description = f'''
        <xpsrfits.Backend>
            name:       {self.name}
            config:     {self.config}
            phase:      {self.phase}
            dcc:        {self.dcc}
            delay:      {self.delay}
            cycle_time: {self.cycle_time}
        '''
        return dedent(description).strip('\n')
