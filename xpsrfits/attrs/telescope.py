from textwrap import dedent
from astropy.coordinates import EarthLocation
import astropy.units as u

class Telescope:
    __slots__ = 'name', 'location'
    
    def __init__(self, name, location):
        self.name = name
        self.location = location
    
    @classmethod
    def from_header(cls, header):
        try:
            ant_x = float(header['ant_x'])
            ant_y = float(header['ant_y'])
            ant_z = float(header['ant_z'])
            location = EarthLocation.from_geocentric(ant_x*u.m, ant_y*u.m, ant_z*u.m)
        except ValueError:
            location = None
        return cls(
            name = header['telescop'],
            location = location,
        )
    
    def __str__(self):
        return f'<{self.name}>'
    
    def __repr__(self):
        description = f'''
        <xpsrfits.Telescope>
            name: {self.name}
            location: {self.location}
        '''
        return dedent(description)[1:]
