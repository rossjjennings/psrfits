import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, Longitude
import astropy.units as u
import dask.array as da
from dask import delayed
from pint import PulsarMJD
from psrfits.formatting import fmt_items, fmt_array
from psrfits.attrs import *
from psrfits.attrs.attrcollection import maybe_missing
from textwrap import indent

class DataFile:
    def __init__(self, data, weights, **attrs):
        self.data = data
        self.weights = weights
        for attr, val in attrs.items():
            setattr(self, attr, val)

    @classmethod
    def from_file(cls, filename, loader='lazy', uniformize_freqs=False):
        '''
        Construct a Observation object from a PSRFITS file.
        '''
        hdulist = fits.open(filename)
        primary_hdu = hdulist['primary']
        history_hdu = hdulist['history']
        subint_hdu = hdulist['subint']

        primary_hdu = hdulist['primary']
        history_hdu = hdulist['history']
        subint_hdu = hdulist['subint']

        # get frequencies
        dat_freq = subint_hdu.data['dat_freq']
        freq = np.atleast_1d(dat_freq[0].copy())
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

        start_time = Time(primary_hdu.header['stt_imjd'], format='pulsar_mjd')
        start_time += primary_hdu.header['stt_smjd']*u.s
        start_time += primary_hdu.header['stt_offs']*u.s

        attrs = {
            'epoch': start_time + subint_hdu.data['offs_sub'].copy()*u.s,
            'freq': freq*u.MHz,
            'phase': phase,
            'duration': subint_hdu.data['tsubint'].copy()*u.s,
            'index': subint_hdu.data['indexval'],
            'source': Source.from_hdulist(hdulist),
            'observation': Observation.from_header(primary_hdu.header),
            'telescope': Telescope.from_header(primary_hdu.header),
            'frontend': Frontend.from_header(primary_hdu.header),
            'backend': Backend.from_header(primary_hdu.header),
            'beam': Beam.from_header(primary_hdu.header),
            'calibrator': Calibrator.from_header(primary_hdu.header),
            'history': History.from_hdu(history_hdu),
            'center_freq': primary_hdu.header['obsfreq']*u.MHz,
            'bandwidth': primary_hdu.header['obsbw']*u.MHz,
            'channel_offset': maybe_missing(subint_hdu.header['nchnoffs']), # *
            'channel_bandwidth': subint_hdu.header['chan_bw']*u.MHz,
            'DM': subint_hdu.header['DM']*u.pc/u.cm**3,
            'RM': subint_hdu.header['RM']*u.rad/u.m**2,
            'n_polns': subint_hdu.header['npol'],
            'pol_type': subint_hdu.header['pol_type'],
            'start_time': start_time,
            'start_lst': Longitude(primary_hdu.header['stt_lst']/3600, u.hourangle),
            'epoch_type': subint_hdu.header['epochs'],
            'time_var': subint_hdu.header['int_type'],
            'time_unit': subint_hdu.header['int_unit'],
            'flux_unit': subint_hdu.header['scale'],
        }

        attrs.update({
            'time_per_bin': history_hdu.data['tbin'][-1],
            'lst': Longitude(subint_hdu.data['lst_sub'].copy()/3600, u.hourangle),
            'coords': SkyCoord(
                subint_hdu.data['ra_sub'].copy(),
                subint_hdu.data['dec_sub'].copy(),
                frame='icrs', unit='deg',
            ),
            'coords_galactic': SkyCoord(
                subint_hdu.data['glon_sub'].copy(),
                subint_hdu.data['glat_sub'].copy(),
                frame='galactic', unit='deg',
            ),
            'feed_angle': subint_hdu.data['fd_ang'].copy(),
            'pos_angle': subint_hdu.data['pos_ang'].copy(),
            'par_angle': subint_hdu.data['par_ang'].copy(),
            'coords_altaz': SkyCoord(
                subint_hdu.data['tel_az'].copy(),
                90 - subint_hdu.data['tel_zen'].copy(),
                frame='altaz', unit='deg',
                obstime=attrs['epoch'], location=attrs['telescope'].location,
            ),
            'aux_dm': subint_hdu.data['aux_dm'].copy()*u.pc/u.cm**3,
            'aux_rm': subint_hdu.data['aux_rm'].copy()*u.rad/u.m**2,
        })
        hdulist.close()

        if loader == 'lazy':
            load = load_delayed
        elif loader == 'mmap':
            load = load_mmap
        elif loader == 'eager':
            load = load_copy

        data = load(filename, 'subint', 'data')
        weights = load(filename, 'subint', 'dat_wts')
        weights = weights.reshape(data.shape[0], data.shape[2])
        scale = load(filename, 'subint', 'dat_scl')
        npol = attrs['n_polns']
        scale = scale.reshape(data.shape[:3])
        offset = load(filename, 'subint', 'dat_offs')
        offset = offset.reshape(data.shape[:3])
        attrs['frequencies'] = load(filename, 'subint', 'dat_freq')

        data = scale[..., np.newaxis]*data + offset[..., np.newaxis]

        pol_type = attrs['pol_type']
        if pol_type in ['AA+BB', 'INTEN']:
            attrs['I'] = data[:,0]
        elif pol_type == 'AABB':
            attrs['AA'] = data[:,0]
            attrs['BB'] = data[:,1]
        elif pol_type == 'AABBCRCI':
            attrs['AA'] = data[:,0]
            attrs['BB'] = data[:,1]
            attrs['CR'] = data[:,2]
            attrs['CI'] = data[:,3]
        elif pol_type == 'IQUV':
            attrs['I'] = data[:,0]
            attrs['Q'] = data[:,1]
            attrs['U'] = data[:,2]
            attrs['V'] = data[:,3]
        else:
            raise ValueError("Polarization type '{}' not recognized.".format(pol_type))

        obs = cls(data, weights, **attrs)

        return obs

    def copy(self):
        attrs = {}
        for attr, value in self.__dict__.items():
            if isinstance(attr, np.ndarray):
                attrs[attr] = value.copy()
            else:
                attrs[attr] = value
        data, weights = attrs['data'], attrs['weights']
        del attrs['data'], attrs['weights']
        return self.__class__(data, weights, **attrs)

    def __repr__(self):
        return (
            f"<psrfits.{self.__class__.__name__}: "
            f"{self.observation.mode} {self.source.name}, "
            f"{self.frontend.name} {self.backend.name} ({self.telescope.name}), "
            f"{self.start_time.iso}"
            ">"
        )

    def info(self):
        nsub, npol, nchan, nbin = self.data.shape
        info_items = {
            'Source': self.source.name,
            'Mode': self.observation.mode,
            'Telescope': self.telescope.name,
            'Frontend': self.frontend.name,
            'Backend': self.backend.name,
            'Project ID': self.observation.project_id,
            'Start Time': self.start_time,
            'Duration': f"{np.sum(self.duration):g} s",
            'Subintegrations': nsub,
            'Center Frequency': self.center_freq,
            'Bandwidth': self.bandwidth,
            'Channels': nchan,
            'Polarizations': self.pol_type,
            'Phase Bins': nbin,
        }
        print(fmt_items(info_items))

    def all_attrs(self):
        print(fmt_items(self.__dict__))

def load_copy(filename, hdu, column):
    '''
    Open a FITS file and make an in-memory copy of a specific HDU column.
    Intended to be used in delayed form by _load_delayed.
    '''
    hdul = fits.open(filename)
    arr = hdul[hdu].data[column].copy()
    hdul.close()
    return arr

def load_delayed(filename, hdu, column):
    '''
    Make a single-chunk Dask array from a column of an HDU within a FITS file,
    using dask.delayed to open and read the file only when needed.
    '''
    hdul = fits.open(filename)
    shape = hdul[hdu].data[column].shape
    dtype = hdul[hdu].data[column].dtype
    hdul.close()
    arr = delayed(load_copy)(filename, hdu, column)

    # hdul[hdu].data[column] goes out of scope here and can be garbage collected.
    # If this were not the case, it would keep around an open file handle.
    return da.from_delayed(arr, shape, dtype, name=f'{filename}.{hdu}.{column}')

def load_memmap(filename, hdu, column, chunks=None):
    '''
    Make a Dask array from a column of an HDU within a memory-mapped FITS file.
    The array will be faster to work with than if _load_delayed() were used, but
    it will keep around an open file handle, and if too many files are read
    this way, you might get a "too many open files" error.
    '''
    hdul = fits.open(filename)
    arr = hdul[hdu].data[col]
    hdul.close()
    # Providing a `name` argument to `da.from_array()` is crucial for performance,
    # since otherwise the name is derived by hashing each chunk.
    return da.from_array(arr, name=f'{filename}.{hdu}.{column}', chunks=chunks)
