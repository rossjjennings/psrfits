import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, Longitude
import astropy.units as u
import dask.array as da
from dask import delayed
from pint import PulsarMJD
from psrfits.attrs import *
from psrfits.attrs.attrcollection import maybe_missing
from psrfits.polyco import PolycoHistory
from psrfits.t2predict import ChebyModelSet
from psrfits.dataset import Dataset
from textwrap import indent
import warnings

class DataFile(Dataset):
    '''
    An object representing a PSRFITS file.
    '''
    @classmethod
    def from_file(cls, filename, loader='memmap', uniformize_freqs=False):
        '''
        Construct a DataFile object from a PSRFITS file.

        Parameters
        ----------
        filename: Name of the file to read data from.
        loader: Possible values are 'copy', 'memmap', 'delayed', 'delayed_memmap', or 'astropy'.
                With 'copy', the data will be read into memory immediately.
                With 'memmap' (the default), the data will be memory mapped and read into
                  a Dask array, retaining a file handle.
                With 'delayed', the data will be read into a Dask array using dask.delayed.
                  This is useful because it does not leave a file handle open, but is slower
                  than the 'memmap' approach when the data are first read.
                With 'delayed_memmap', the data will be memory mapped and read into a Dask
                  array using dask.delayed, which should not leave a file handle open.
                With 'astropy', the memory mapping internal to astropy.io.fits will be used.
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

        start_time = Time(primary_hdu.header['stt_imjd'], scale='utc', format='pulsar_mjd')
        start_time += primary_hdu.header['stt_smjd']*u.s
        start_time += primary_hdu.header['stt_offs']*u.s

        out = cls()

        # coordinates
        out.epoch =  start_time + subint_hdu.data['offs_sub'].copy()*u.s
        out.freq = freq*u.MHz
        out.phase = phase

        # special values
        if 'polyco' in hdulist:
            out.polyco = PolycoHistory(hdulist['polyco'].data)
        if 't2predict' in hdulist:
            out.predictor = ChebyModelSet.parse(
                [line[0] for line in hdulist['t2predict'].data]
            )
        if 'psrparam' in hdulist:
            out.model = '\n'.join(line[0] for line in hdulist['psrparam'].data)
        out.observation = Observation.from_header(primary_hdu.header)
        out.telescope = Telescope.from_header(primary_hdu.header)
        out.frontend = Frontend.from_header(primary_hdu.header)
        out.backend = Backend.from_header(primary_hdu.header)
        out.beam = Beam.from_header(primary_hdu.header)
        out.calibrator = Calibrator.from_header(primary_hdu.header)
        out.history = History.from_hdu(history_hdu)

        # scalar attributes
        out.source = primary_hdu.header['src_name']
        out.start_time = start_time
        out.center_freq = primary_hdu.header['obsfreq']*u.MHz
        out.bandwidth = primary_hdu.header['obsbw']*u.MHz
        out.channel_offset = maybe_missing(subint_hdu.header['nchnoffs']) # *
        out.channel_bandwidth = subint_hdu.header['chan_bw']*u.MHz
        out.DM = subint_hdu.header['DM']*u.pc/u.cm**3
        out.RM = subint_hdu.header['RM']*u.rad/u.m**2
        out.n_polns = subint_hdu.header['npol']
        out.pol_type = subint_hdu.header['pol_type']
        out.start_lst = Longitude(primary_hdu.header['stt_lst']/3600, u.hourangle)
        out.epoch_type = subint_hdu.header['epochs']
        out.time_var = subint_hdu.header['int_type']
        out.time_unit = subint_hdu.header['int_unit']
        out.flux_unit = subint_hdu.header['scale']

        # time serries attributes
        out.duration = subint_hdu.data['tsubint'].copy()*u.s
        out.index = subint_hdu.data['indexval']
        field_names = subint_hdu.data.dtype.names
        if 'LST_SUB' in field_names:
            out.lst = Longitude(subint_hdu.data['lst_sub'].copy()/3600, u.hourangle)
        if 'RA_SUB' in field_names and 'DEC_SUB' in field_names:
            out.coords = SkyCoord(
                subint_hdu.data['ra_sub'].copy(),
                subint_hdu.data['dec_sub'].copy(),
                frame='icrs', unit='deg',
            )
        if 'GLON_SUB' in field_names and 'GLAT_SUB' in field_names:
            out.coords_galactic = SkyCoord(
                subint_hdu.data['glon_sub'].copy(),
                subint_hdu.data['glat_sub'].copy(),
                frame='galactic', unit='deg',
            )
        if 'FD_ANG' in field_names:
            out.feed_angle = subint_hdu.data['fd_ang'].copy()
        if 'POS_ANG' in field_names:
            out.pos_angle = subint_hdu.data['pos_ang'].copy()
        if 'PAR_ANG' in field_names:
            out.par_angle = subint_hdu.data['par_ang'].copy()
        if 'TEL_AZ' in field_names and 'TEL_ZEN' in field_names:
            out.coords_altaz = SkyCoord(
                subint_hdu.data['tel_az'].copy(),
                90 - subint_hdu.data['tel_zen'].copy(),
                frame='altaz', unit='deg',
                obstime=out.epoch, location=out.telescope.location,
            )
        out.aux_dm = subint_hdu.data['aux_dm'].copy()*u.pc/u.cm**3
        out.aux_rm = subint_hdu.data['aux_rm'].copy()*u.rad/u.m**2
        hdulist.close()

        if loader == 'copy':
            load = load_copy
        elif loader == 'delayed':
            load = load_delayed
        elif loader == 'astropy':
            load = load_astropy
        elif loader == 'memmap':
            load = load_memmap
        elif loader == 'delayed_memmap':
            load = load_delayed_memmap

        data = load(filename, 'subint', 'data', chunks=('auto', 'auto', 'auto', -1))
        scale = load(filename, 'subint', 'dat_scl')
        scale = scale.reshape(data.shape[:3])
        offset = load(filename, 'subint', 'dat_offs')
        offset = offset.reshape(data.shape[:3])

        weights = load(filename, 'subint', 'dat_wts')
        weights = weights.reshape(data.shape[0], data.shape[2])
        out.weights = weights
        out.frequencies = load(filename, 'subint', 'dat_freq')

        data = scale[..., np.newaxis]*data + offset[..., np.newaxis]

        if out.pol_type in ['AA+BB', 'INTEN']:
            out.I = data[:,0]
        elif out.pol_type == 'AABB':
            out.AA = data[:,0]
            out.BB = data[:,1]
        elif out.pol_type == 'AABBCRCI':
            out.AA = data[:,0]
            out.BB = data[:,1]
            out.CR = data[:,2]
            out.CI = data[:,3]
        elif out.pol_type == 'IQUV':
            out.I = data[:,0]
            out.Q = data[:,1]
            out.U = data[:,2]
            out.V = data[:,3]
        else:
            raise ValueError("Polarization type '{}' not recognized.".format(pol_type))

        return out

def load_copy(filename, hdu, column, chunks="auto"):
    '''
    Open a FITS file and make an in-memory copy of a specific HDU column.
    Intended to be used in delayed form by _load_delayed.
    '''
    hdul = fits.open(filename)
    arr = hdul[hdu].data[column].copy()
    hdul.close()
    return arr

def load_delayed(filename, hdu, column, chunks="auto"):
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

def load_astropy(filename, hdu, column, chunks="auto"):
    '''
    Make a Dask array from a column of an HDU within a memory-mapped FITS file.
    This version uses Astropy's internal mechanisms to manage the memory map.
    The array will be faster to work with than if _load_delayed() were used, but
    it will keep around an open file handle, and if too many files are read
    this way, you might get a "too many open files" error.
    '''
    hdul = fits.open(filename)
    arr = hdul[hdu].data[column]
    hdul.close()
    # Providing a `name` argument to `da.from_array()` is crucial for performance,
    # since otherwise the name is derived by hashing each chunk.
    return da.from_array(arr, name=f'{filename}.{hdu}.{column}', chunks=chunks)

def load_memmap(filename, hdu, column, chunks="auto"):
    '''
    Make a Dask array from a column of an HDU within a fits file. Uses Astropy to
    collect initial information about the location of data within the file, and then
    constructs a memory map separately. This should not leave dangling Astropy objects.
    However, because of the memory map, it will still leave a file open.
    '''
    hdul = fits.open(filename)
    data_dtype = hdul[hdu].data.dtype
    data_shape = hdul[hdu].data.shape
    offset = hdul[hdu].fileinfo()['datLoc']
    hdul.close()
    data = np.memmap(filename, mode='r', offset=offset, dtype=data_dtype, shape=data_shape)
    return da.from_array(data[column.upper()], name=f'{filename}.{hdu}.{column}', chunks=chunks)

def load_delayed_memmap(filename, hdu, column, chunks="auto"):
    '''
    Make a Dask array from a column of an HDU within a fits file. This is a combination
    of `load_delayed` and `load_memmap`: It returns a Dask array which is based on a delayed
    numpy memmap. In theory, this should allow some of the speed advantages of memmap, without
    keeping a file open.
    '''
    hdul = fits.open(filename)
    data_dtype = hdul[hdu].data.dtype
    data_shape = hdul[hdu].data.shape
    offset = hdul[hdu].fileinfo()['datLoc']
    hdul.close()
    data = np.memmap(filename, mode='r', offset=offset, dtype=data_dtype, shape=data_shape)
    return da.from_array(data[column.upper()], name=f'{filename}.{hdu}.{column}', chunks=chunks)
