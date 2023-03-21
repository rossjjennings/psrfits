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
from psrfits import baseline, dispersion, polarization, plots
from textwrap import indent

class DataFile:
    '''
    An object representing a PSRFITS file.
    '''
    def __init__(self, data, weights, **attrs):
        self.data = data
        self.weights = weights
        for attr, val in attrs.items():
            setattr(self, attr, val)

    @classmethod
    def from_file(cls, filename, loader='lazy', uniformize_freqs=False):
        '''
        Construct a DataFile object from a PSRFITS file.

        Parameters
        ----------
        filename: Name of the file to read data from.
        loader: Possible values are 'lazy', 'memmap', or 'eager'. For 'lazy' (the default),
                the data will be read into a Dask array using dask.delayed, and read only
                when necessary. No file handle will be retained. For 'memmap', the data will
                be memory-mapped and read into a Dask array, retaining a file handle. For
                'eager', the data will be read into memory immediately.
        uniformize_freqs: Whether to attempt to correct over-rounded frequency values.
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
        '''
        Make a copy of this DataFile. This is a "deep copy" in the sense that
        all attributes are copied, even those that are Numpy or Dask arrays, but
        Dask arrays will continue to point to the same data on disk.
        '''
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
        '''
        Print a summary of the data contained in this object.
        '''
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
        '''
        Print a formatted list of all attributes of this object.
        '''
        print(fmt_items(self.__dict__))

    def dedisperse(self, DM=None, weight_center_freq=False):
        '''
        Dedisperse the data with the given DM.
        If `DM` is `None`, use the DM attribute of `ds`.
        '''
        dispersion.dedisperse(self, inplace=True, DM=DM, weight_center_freq=weight_center_freq)

    def align_with_predictor(self, out_of_bounds='error'):
        '''
        Dedisperse and align the data using the internal Tempo2 predictor.
        '''
        dispersion.align_with_predictor(self, inplace=True, out_of_bounds=out_of_bounds)

    def remove_baseline(self, method='avgprof', frac=1/8):
        '''
        Remove the frequency-dependent baseline from an observation.

        Parameters
        ----------
        method (default: 'avgprof'): The method used to determine the baseline level.
            Options are 'avgprof', which takes the mean of an "off-pulse" region
            automatically determined from the average total-intensity profile,
            'offpulse', which is uses the off-pulse mean for each channel separately,
            and 'median', which takes the median of the entire profile in each channel
            separately.
        frac (default: 1/8): The fraction of the profile to use as an off-pulse window.
        '''
        baseline.remove_baseline(self, inplace=True, method=method, frac=frac)

    def get_pols(self):
        '''
        Return a list of the polarizations present in the data.
        '''
        polarization.get_pols(self)

    def pscrunch(self):
        '''
        Return a dataset containing only the total intensity component of the input.
        '''
        polarization.pscrunch(self, inplace=True)

    def to_stokes(self):
        '''
        Transform coherence (AABBCRCI) data to Stokes parameters (IQUV).
        If input is already Stokes, leave it alone.
        If input has one or two polarizations, return I only.
        '''
        polarization.to_stokes(self, inplace=True)

    def plot_profile(self, profile, ax=None, **kwargs):
        '''
        Make a line plot of a supplied pulse profile using metadata from this
        Dataset. Additional keyword arguments will be passed on to plt.plot().

        Parameters
        ----------
        ds:      Dataset to use
        profile: Profile (array of data vs. pulse phase) to plot
        ax:      Axes in which to plot. If `None`, the current Axes will be used.
        '''
        plots.plot_profile(self, profile, ax, **kwargs)

    def plot_portrait(self, profile, ax=None, sym_lim=False, vmin=None, vmax=None, **kwargs):
        '''
        Make a pseudocolor plot of a supplied pulse portrait (pulse phase vs. frequency)
        using metadata from this Dataset. Additional keyword arguments will be passed
        on to plt.pcolormesh().

        Parameters
        ----------
        portrait: Portrait (array of data vs. frequency and phase) to plot
        ax:       Axes in which to plot. If `None`, the current Axes will be used.
        sym_lim:  Symmetrize the colorbar limits around zero. Useful when plotting
                  signed data using a diverging colormap.
        '''
        plots.plot_portrait(self, profile, ax, sym_lim, vmin, vmax, **kwargs)

    def plot_pulsetrain(self, pulsetrain, ax=None, sym_lim=False, vmin=None, vmax=None, **kwargs):
        '''
        Make a pseudocolor plot of a supplied "pulse train" (i.e., time series of profiles
        matching the length of the underlying data, a bit of a misnomer) using metadata
        from this Dataset. Additional keyword arguments will be passed on to plt.pcolormesh().

        Parameters
        ----------
        ds:      Dataset to use
        profile: Profile (array of data vs. pulse phase) to plot
        ax:      Axes in which to plot. If `None`, the current Axes will be used.
        sym_lim: Symmetrize the colorbar limits around zero. Useful when plotting
                 signed data using a diverging colormap.
        '''
        plots.plot_pulsetrain(self, pulsetrain, ax, sym_lim, vmin, vmax, **kwargs)

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
