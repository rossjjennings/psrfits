import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, Longitude
import astropy.units as u
from psrfits.formatting import fmt_items, fmt_array
from psrfits.attrs import *
from psrfits.attrs.attrcollection import maybe_missing
from textwrap import indent

class DataFile:
    def __init__(self, data, **attrs):
        self.data = data
        for attr, val in attrs.items():
            setattr(self, attr, val)

    @classmethod
    def from_file(cls, filename, uniformize_freqs=False):
        '''
        Construct a Observation object from a PSRFITS file.
        '''
        hdulist = fits.open(filename)
        primary_hdu = hdulist['primary']
        history_hdu = hdulist['history']
        subint_hdu = hdulist['subint']

        data = subint_hdu.data['data']
        #data_vars = pol_split(data, subint_hdu.header['pol_type'])

        primary_hdu = hdulist['primary']
        history_hdu = hdulist['history']
        subint_hdu = hdulist['subint']

        # Get coordinates
        time = subint_hdu.data['offs_sub']
        dat_freq = subint_hdu.data['dat_freq']
        freq = np.atleast_1d(dat_freq[0])
        channel_bandwidth = subint_hdu.header['chan_bw']
        bandwidth = primary_hdu.header['obsbw']

        if uniformize_freqs:
            # Undo effects of truncation to nearest 1 kHz (introduced by PSRCHIVE)
            freq = uniformize(freq, channel_bandwidth)
            # (Uniformized) first row should match everything to within 1 kHz
            if np.any(np.abs(dat_freq - freq) > 0.001):
                msg = 'Not all frequencies match within tolerance.'
                warnings.warn(msg, RuntimeWarning)

        # Difference between first and last frequencies should equal bandwidth
        discrepancy = np.abs(freq[-1] - freq[0] + channel_bandwidth - bandwidth)
        if discrepancy > 0.001 or (uniformize_freqs and discrepancy != 0):
            msg = 'Frequencies do not match bandwidth. Band edges may be missing.'
            warnings.warn(msg, RuntimeWarning)

        nbin = history_hdu.data['nbin'][-1]
        try:
            nbin_prd = int(history_hdu.data['nbin_prd'][-1])
        except ValueError:
            nbin_prd = nbin
        try:
            phs_offs = float(subint_hdu.header['phs_offs'])
        except ValueError:
            phs_offs = 0.
        phase = np.linspace(0., nbin/nbin_prd, nbin, endpoint=False) + phs_offs

        start_time = Time(primary_hdu.header['stt_imjd'], format='mjd')
        start_time.format = 'isot'
        start_time += primary_hdu.header['stt_smjd']*u.s
        start_time += primary_hdu.header['stt_offs']*u.s

        attrs = {
            'time': time,
            'freq': freq,
            'phase': phase,
            'duration': subint_hdu.data['tsubint'],
            'weights': subint_hdu.data['dat_wts'].reshape(time.size, freq.size),
            'index': subint_hdu.data['indexval'],
            'source': Source.from_hdulist(hdulist),
            'observation': Observation.from_header(primary_hdu.header),
            'telescope': Telescope.from_header(primary_hdu.header),
            'frontend': Frontend.from_header(primary_hdu.header),
            'backend': Backend.from_header(primary_hdu.header),
            'beam': Beam.from_header(primary_hdu.header),
            'calibrator': Calibrator.from_header(primary_hdu.header),
            'history': History.from_hdu(history_hdu),
            'frequency': primary_hdu.header['obsfreq'],
            'bandwidth': primary_hdu.header['obsbw'],
            'center_freq': history_hdu.data['ctr_freq'][-1],
            'channel_offset': maybe_missing(subint_hdu.header['nchnoffs']), # *
            'channel_bandwidth': subint_hdu.header['chan_bw'],
            'DM': subint_hdu.header['DM'],
            'RM': subint_hdu.header['RM'],
            'n_polns': subint_hdu.header['npol'],
            'pol_type': subint_hdu.header['pol_type'],
            'start_time': start_time,
            'start_lst': Longitude(primary_hdu.header['stt_lst']/3600, u.hourangle),
            'epoch_type': subint_hdu.header['epochs'],
            'time_var': subint_hdu.header['int_type'],
            'time_unit': subint_hdu.header['int_unit'],
            'flux_unit': subint_hdu.header['scale'],
            'time_per_bin': history_hdu.data['tbin'][-1],
            'scale': subint_hdu.data['dat_scl'],
            'offset': subint_hdu.data['dat_offs'],
            'lst': subint_hdu.data['lst_sub'],
            'ra': subint_hdu.data['ra_sub'],
            'dec': subint_hdu.data['dec_sub'],
            'glon': subint_hdu.data['glon_sub'],
            'glat': subint_hdu.data['glat_sub'],
            'feed_angle': subint_hdu.data['fd_ang'],
            'pos_angle': subint_hdu.data['pos_ang'],
            'par_angle': subint_hdu.data['par_ang'],
            'az': subint_hdu.data['tel_az'],
            'zen': subint_hdu.data['tel_zen'],
            'aux_dm': subint_hdu.data['aux_dm'],
            'aux_rm': subint_hdu.data['aux_rm'],
        }

        obs = cls(data, **attrs)
        hdulist.close()

        return obs

    def __repr__(self):
        return (
            f"<psrfits.{self.__class__.__name__}: "
            f"{self.observation.mode} {self.source.name}, "
            f"{self.frontend.name} {self.backend.name} ({self.telescope.name}), "
            f"{self.start_time.iso}"
            ">"
        )

    def info(self):
        info_items = {
            'Dimensions': f"(time: {self.time.size}, freq: {self.freq.size}, phase: {self.phase.size})",
            'Source': self.source.name,
            'Mode': self.observation.mode,
            'Telescope': self.telescope.name,
            'Frontend': self.frontend.name,
            'Backend': self.backend.name,
            'Start Time': self.start_time,
            'Duration': f"{np.sum(self.duration):.2f} s",
            'Frequency': self.frequency,
            'Bandwidth': self.bandwidth,
            'Polarizations': self.pol_type,
        }
        print(fmt_items(info_items))

    def all_attrs(self):
        print(fmt_items(self.__dict__))

def fmt_coord(name, coord):
    dims_str = f"({name})"
    dims_dtype = dims_str + ' ' + str(coord.dtype) + ' '
    return dims_dtype + fmt_array(coord, 65 - len(dims_dtype))

def fmt_datavar(datavar):
    dims, var = datavar
    dims_str = f"({', '.join(dims)})"
    dims_dtype = dims_str + ' ' + str(var.dtype) + ' '
    return dims_dtype + fmt_array(var, 65 - len(dims_dtype))
