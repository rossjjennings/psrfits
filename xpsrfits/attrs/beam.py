from textwrap import dedent

class Beam:
    __slots__ = 'beam_id', 'center_id', 'major_axis', 'minor_axis', 'pos_angle'
    
    def __init__(self, beam_id, center_id, major_axis, minor_axis, pos_angle):
        self.beam_id = beam_id
        self.center_id = center_id
        self.major_axis = major_axis
        self.minor_axis = minor_axis
        self.pos_angle = pos_angle
    
    @classmethod
    def from_header(cls, header):
        return cls(
            beam_id = header['ibeam'],
            center_id = header['pnt_id'],
            major_axis = header['bmaj'],
            minor_axis = header['bmin'],
            pos_angle = header['bpa'],
        )
    
    def __str__(self):
        if self.beam_id is None or self.beam_id == '':
            return '<Beam>'
        else:
            return f'<Beam {self.beam_id}>'
    
    def __repr__(self):
        description = f'''
        <xpsrfits.Beam>
            beam_id:    {self.beam_id}
            center_id:  {self.center_id}
            major_axis: {self.major_axis}
            minor_axis: {self.minor_axis}
            pos_angle:  {self.pos_angle}
        '''
        return dedent(description).strip('\n')
