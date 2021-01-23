from textwrap import dedent

class Frontend:
    __slots__ = 'name', 'n_pol', 'feed_poln', 'handedness', 's_angle', 'xy_phase'
    
    def __init__(self, name, n_pol, feed_poln, handedness, s_angle, xy_phase):
        self.name = name
        self.n_pol = n_pol
        self.feed_poln = feed_poln
        self.handedness = handedness
        self.s_angle = s_angle
        self.xy_phase = xy_phase
    
    @classmethod
    def from_header(cls, header):
        return cls(
            name = header['frontend'],
            n_pol = header['nrcvr'],
            feed_poln = header['fd_poln'],
            handedness = header['fd_hand'],
            s_angle = header['fd_sang'],
            xy_phase = header['fd_xyph'],
        )
    
    def __str__(self):
        return f'<{self.name}>'
    
    def __repr__(self):
        description = f'''
        <xpsrfits.Frontend>
            name:       {self.name}
            n_pol:      {self.n_pol}
            feed_poln:  {self.feed_poln}
            handedness: {self.handedness}
            s_angle:    {self.s_angle}
            xy_phase:   {self.xy_phase}
        '''
        return dedent(description).strip('\n')
