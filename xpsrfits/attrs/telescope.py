from textwrap import indent
from astropy.coordinates import EarthLocation
import astropy.units as u

from .attrcollection import AttrCollection

class Telescope(AttrCollection):
    __slots__ = 'name', 'location'
    
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
        description = "<xpsrfits.Telescope>\n"
        description += indent(self._repr_items(), '    ')
        return description
    
    def header_cards(self):
        return {
            'telescop': self.name,
            'ant_x': self.location.x.to(u.m).value,
            'ant_y': self.location.y.to(u.m).value,
            'ant_z': self.location.z.to(u.m).value,
        }
