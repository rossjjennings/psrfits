from textwrap import indent
from astropy.coordinates import EarthLocation
import astropy.units as u

class Telescope:
    __slots__ = 'name', 'location'
    
    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs[name])
    
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
    
    def _repr_items(self):
        max_len = max(len(name) for name in self.__slots__)
        description = ""
        for name in self.__slots__:
            key = f"{name}:"
            description += f"{key:<{max_len + 2}}{getattr(self, name)}\n"
        return description
    
    def __str__(self):
        return f'<{self.name}>'
    
    def __repr__(self):
        description = "<xpsrfits.Telescope>\n"
        description += indent(self._repr_items(), '    ')
        return description
