from textwrap import dedent
from astropy.time import Time

class Observation:
    __slots__ = 'date', 'observer', 'mode', 'project_id', 'coords', 'track_mode'
    
    def __init__(self, date, observer, mode, project_id, coords, track_mode):
        self.date = date
        self.observer = observer
        self.mode = mode
        self.project_id = project_id
        self.coords = coords
        self.track_mode = track_mode
    
    @classmethod
    def from_header(cls, header):
        return cls(
            date = Time(header['date-obs']),
            observer = header['observer'],
            mode = header['obs_mode'],
            project_id = header['projid'],
            coords = header['coord_md'],
            track_mode = header['trk_mode'],
        )
    
    def __str__(self):
        return f'{self.mode} mode observation [...]'
    
    def __repr__(self):
        description = f'''
        <xpsrfits.Observation>
            date:       {self.date}
            observer:   {self.observer}
            mode:       {self.mode}
            project_id: {self.project_id}
            coords:     {self.coords}
            track_mode: {self.track_mode}
        '''
        return dedent(description).strip('\n')
