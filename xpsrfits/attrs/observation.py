from textwrap import indent
from astropy.time import Time
from astropy.coordinates import SkyCoord, ICRS
import astropy.units as u
from pint import PulsarEcliptic
from datetime import datetime
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
        date = maybe_missing(header['date-obs'])
        if date is not None:
            print(date)
            date = Time(date)
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
            start_ra = maybe_missing(header['stt_crd1'])
            start_dec = maybe_missing(header['stt_crd2'])
            if start_ra is not None and start_dec is not None:
                start_coords = SkyCoord(
                    ra=start_ra,
                    dec=start_ra,
                    unit=(u.hourangle, u.deg),
                    frame='icrs',
                )
            else:
                start_coords = None
            stop_ra = maybe_missing(header['stp_crd1'])
            stop_dec = maybe_missing(header['stp_crd2'])
            if stop_ra is not None and stop_dec is not None:
                stop_coords = SkyCoord(
                    ra=stop_ra,
                    dec=stop_dec,
                    unit=(u.hourangle, u.deg),
                    frame='icrs',
                )
            else:
                stop_coords = None
        elif coord_mode == 'GALACTIC':
            warnings.warn("Interpreting galactic coordinates using astropy.coordinates.Galactic")
            start_lon = maybe_missing(header['stt_crd1'])
            start_lat = maybe_missing(header['stt_crd2'])
            if start_lon is not None and start_lat is not None:
                start_coords = SkyCoord(
                    lon=start_lon,
                    lat=start_lat,
                    unit=(u.deg, u.deg),
                    frame='galactic',
                )
            else:
                start_coords = None
            stop_lon = maybe_missing(header['stp_crd1'])
            stop_lat = maybe_missing(header['stp_crd2'])
            if stop_lon is not None and stop_lat is not None:
                stop_coords = SkyCoord(
                    lon=stop_lon,
                    lat=stop_lat,
                    unit=(u.deg, u.deg),
                    frame='galactic',
                )
            else:
                stop_coords = None
        elif coord_mode == 'ECLIPTIC':
            warnings.warn("Interpreting ecliptic coordinates using IERS2010 obliquity")
            start_lon = maybe_missing(header['stt_crd1'])
            start_lat = maybe_missing(header['stp_crd1'])
            if start_lat is not None and start_lon is not None:
                start_coords = SkyCoord(
                    lon=start_lon,
                    lat=start_lat,
                    unit=(u.deg, u.deg),
                    frame='pulsarecliptic',
                )
            else:
                start_coords = None
            stop_lon = maybe_missing(header['stp_crd1'])
            stop_lat = maybe_missing(header['stp_crd2'])
            if stop_lon is not None and start_lon is not None:
                stop_coords = SkyCoord(
                    lon=stop_lon,
                    lat=stop_lat,
                    unit=(u.deg, u.deg),
                    frame='pulsarecliptic',
                )
            else:
                stop_coords = None
        
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
        
        if self.start_coords is None:
            start_coord1_str = 'UNSET'
            start_coord2_str = 'UNSET'
        else:
            start_coords_icrs = self.start_coords.transform_to(ICRS)
            start_coord1_str = start_coords_icrs.ra.to_string(unit=u.hourangle, sep=':')
            start_coord2_str = start_coords_icrs.dec.to_string(unit=u.deg, sep=':')
        
        if self.stop_coords is None:
            stop_coord1_str = 'UNSET'
            stop_coord2_str = 'UNSET'
        else:
            stop_coords_icrs = self.stop_coords.transform_to(ICRS)
            stop_coord1_str = stop_coords_icrs.ra.to_string(unit=u.hourangle, sep=':')
            stop_coord2_str = stop_coords_icrs.dec.to_string(unit=u.deg, sep=':')
        
        if self.date is None:
            date_str = 'UNSETTUNSET'
        else:
            date_str = datetime.strftime(self.date.datetime, '%Y-%m-%dT%H:%M:%S')
        
        return {
            'observer': self.observer,
            'projid': self.project_id,
            'obs_mode': self.mode,
            'date-obs': date_str,
            'coord_md': 'J2000',
            'equinox': '2000.0',
            'ra': coords_icrs.ra.to_string(unit=u.hourangle, sep=':'),
            'dec': coords_icrs.dec.to_string(unit=u.deg, sep=':'),
            'stt_crd1': start_coord1_str,
            'stt_crd2': start_coord2_str,
            'trk_mode': self.track_mode,
            'stp_crd1': stop_coord1_str,
            'stp_crd2': stop_coord2_str,
            'scanlen': if_missing('*', self.scan_length),
            'fd_mode': self.feed_mode,
            'fa_req': self.feed_angle,
        }
