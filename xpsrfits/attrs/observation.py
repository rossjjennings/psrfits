from textwrap import indent
from astropy.time import Time
from astropy.coordinates import SkyCoord, ICRS
import astropy.units as u
from pint import PulsarEcliptic
import warnings

from .attrcollection import AttrCollection, maybe_missing, if_missing

class Observation(AttrCollection):
    __slots__ = (
        'date',
        'observer',
        'mode',
        'project_id',
        'coords',
        'track_mode',
        'start_coords',
        'stop_coords',
        'scan_length',
        'feed_mode',
        'feed_angle'
    )
    
    @classmethod
    def from_header(cls, header):
        date = Time(header['date-obs'])
        coord_mode = header['coord_md']
        equinox = header['equinox']
        
        if float(equinox) == 2000.0:
            coords = SkyCoord(
                ra=header['ra'],
                dec=header['dec'],
                unit=(u.hourangle, u.deg),
                frame='icrs',
            )
        else:
            msg = f"Interpreting equinox J{equinox} using astropy.coordinates.PrecessedGeocentric"
            warnings.warn(msg, RuntimeWarning)
            coords = SkyCoord(
                ra=header['ra'],
                dec=header['dec'],
                unit=(u.hourangle, u.deg),
                frame='precessedgeocentric',
                equinox=Time(equinox, format='jyear'),
                obstime=date,
            )
        
        if coord_mode == 'J2000':
            start_coords = SkyCoord(
                ra=header['stt_crd1'],
                dec=header['stt_crd2'],
                unit=(u.hourangle, u.deg),
                frame='icrs',
            )
            stop_coords = SkyCoord(
                ra=header['stp_crd1'],
                dec=header['stp_crd2'],
                unit=(u.hourangle, u.deg),
                frame='icrs',
            )
        elif coord_mode == 'GALACTIC':
            warnings.warn("Interpreting galactic coordinates using astropy.coordinates.Galactic")
            start_coords = SkyCoord(
                lon=header['stt_crd1'],
                lat=header['stt_crd2'],
                unit=(u.deg, u.deg),
                frame='galactic',
            )
            stop_coords = SkyCoord(
                lon=header['stp_crd1'],
                lat=header['stp_crd2'],
                unit=(u.deg, u.deg),
                frame='galactic',
            )
        elif coord_mode == 'ECLIPTIC':
            warnings.warn("Interpreting ecliptic coordinates using IERS2010 obliquity")
            start_coords = SkyCoord(
                lon=header['stt_crd1'],
                lat=header['stt_crd2'],
                unit=(u.deg, u.deg),
                frame='pulsarecliptic',
            )
            stop_coords = SkyCoord(
                lon=header['stp_crd1'],
                lat=header['stp_crd2'],
                unit=(u.deg, u.deg),
                frame='pulsarecliptic',
            )
        
        return cls(
            date = date,
            observer = header['observer'],
            mode = header['obs_mode'],
            project_id = header['projid'],
            coords = coords,
            track_mode = header['trk_mode'],
            start_coords = stop_coords,
            stop_coords = stop_coords,
            scan_length = maybe_missing(header['scanlen']),
            feed_mode = header['fd_mode'],
            feed_angle = header['fa_req'],
        )
    
    def __str__(self):
        return f'<{self.mode} mode observation>'
    
    def __repr__(self):
        description = "<xpsrfits.Observation>\n"
        description += indent(self._repr_items(), '    ')
        return description
    
    def header_cards(self):
        if (self.coords.frame.__class__.__name__ != 'ICRS'
            or self.start_coords.frame.__class__.__name__ != 'ICRS'
            or self.stop_coords.frame.__class__.__name__ != 'ICRS'):
            warnings.warn("Converting coordinates to J2000 for output")
        coords_icrs = self.coords.transform_to(ICRS)
        start_coords_icrs = self.start_coords.transform_to(ICRS)
        stop_coords_icrs = self.stop_coords.transform_to(ICRS)
        return {
            'observer': self.observer,
            'projid': self.project_id,
            'obs_mode': self.mode,
            'date-obs': self.date.isot,
            'coord_md': 'J2000',
            'equinox': '2000.0',
            'ra': coords_icrs.ra.to_string(unit=u.hourangle, sep=':'),
            'dec': coords_icrs.dec.to_string(unit=u.deg, sep=':'),
            'stt_crd1': start_coords_icrs.ra.to_string(unit=u.hourangle, sep=':'),
            'stt_crd2': start_coords_icrs.dec.to_string(unit=u.deg, sep=':'),
            'trk_mode': self.track_mode,
            'stp_crd1': stop_coords_icrs.ra.to_string(unit=u.hourangle, sep=':'),
            'stp_crd2': stop_coords_icrs.dec.to_string(unit=u.deg, sep=':'),
            'scanlen': if_missing('*', self.scan_length),
            'fd_mode': self.feed_mode,
            'fa_req': self.feed_angle,
        }
