from textwrap import dedent
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from pint import PulsarEcliptic
import warnings

class Observation:
    __slots__ = 'date', 'observer', 'mode', 'project_id', 'coords', 'track_mode', 'start_coords', 'stop_coords', 'scan_length', 'feed_mode', 'feed_angle'
    
    def __init__(self, date, observer, mode, project_id, coords, track_mode, start_coords, stop_coords, scan_length, feed_mode, feed_angle):
        self.date = date
        self.observer = observer
        self.mode = mode
        self.project_id = project_id
        self.coords = coords
        self.track_mode = track_mode
        self.start_coords = start_coords
        self.stop_coords = stop_coords
        self.scan_length = scan_length
        self.feed_mode = feed_mode
        self.feed_angle = feed_angle
    
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
            msg = f"Equinox is not J2000: using dynamical equinox J{equinox}"
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
            warnings.warn("ECLIPTIC mode is ambiguous: using IERS2010 obliquity")
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
            scan_length = header['scanlen'],
            feed_mode = header['fd_mode'],
            feed_angle = header['fa_req'],
        )
    
    def __str__(self):
        return f'<{self.mode} mode observation>'
    
    def __repr__(self):
        description = f'''
        <xpsrfits.Observation>
            date:         {self.date}
            observer:     {self.observer}
            mode:         {self.mode}
            project_id:   {self.project_id}
            coords:       {fmt_skycoord(self.coords)}
            track_mode:   {self.track_mode}
            start_coords: {fmt_skycoord(self.start_coords)}
            stop_coords:  {fmt_skycoord(self.stop_coords)}
            scan_length:  {self.scan_length}
            feed_mode:    {self.feed_mode}
            feed_angle:   {self.feed_angle}
        '''
        return dedent(description).strip('\n')

def fmt_skycoord(skycoord):
    class_name = skycoord.__class__.__name__
    frame_name = skycoord.frame.__class__.__name__
    component_names = list(skycoord.representation_component_names)
    if 'distance' in component_names and skycoord.distance == 1.0:
        component_names.remove('distance')
    values = ', '.join(f'{name}={getattr(skycoord, name):g}' for name in component_names)
    return f'<{class_name} ({frame_name}): {values}>'
