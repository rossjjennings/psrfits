import numpy as np
import xarray as xr
import warnings
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, Galactic
import astropy.units as u
from datetime import datetime
from textwrap import dedent
import toml
import os.path

from .attrs.attrcollection import if_missing
from .polarization import get_pols

def to_hdulist(ds):
    '''
    Convert an xpsrfits dataset to a FITS HDU list for saving.
    '''
    hdus = []
    hdus.append(construct_primary_hdu(ds))
    hdus.append(construct_history_hdu(ds))
    hdus.append(construct_psrparam_hdu(ds))
    if ds.source.predictor is not None:
        hdus.append(construct_t2predict_hdu(ds))
    if ds.source.polyco is not None:
        hdus.append(construct_polyco_hdu(ds))
    hdus.append(construct_subint_hdu(ds))
    return fits.HDUList(hdus)

def construct_primary_hdu(ds):
    '''
    Construct primary HDU
    '''
    primary_hdu = fits.PrimaryHDU()
    fits_description = dedent("""
    FITS (Flexible Image Transport System) format is defined in 'Astronomy
    and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H.
    Contact the NASA Science Office of Standards and Technology for the
    FITS Definition document #100 and other FITS information.
    """)[1:]
    for line in fits_description.split('\n'):
        primary_hdu.header['comment'] = line
    
    imjd = int(ds.start_time.mjd)
    smjd = (ds.start_time - Time(imjd, format='mjd')).to(u.s).value
    header_cards = {
        'fitstype': 'PSRFITS',
        'hdrver': '6.1',
        'date': datetime.strftime(datetime.now(), '%Y-%m-%dT%H:%M:%S'),
        'obsfreq': ds.frequency,
        'obsbw': ds.bandwidth,
        'obsnchan': ds.history[0].n_channels,
        'chan_dm': ds.DM,
        'src_name': ds.source.name,
        'stt_imjd': imjd,
        'stt_smjd': int(smjd),
        'stt_offs': smjd % 1,
        'stt_lst': 3600*ds.start_lst.to(u.hourangle).value,
    }
    
    header_cards.update(ds.source.header_cards())
    header_cards.update(ds.observation.header_cards())
    header_cards.update(ds.telescope.header_cards())
    header_cards.update(ds.frontend.header_cards())
    header_cards.update(ds.backend.header_cards())
    header_cards.update(ds.beam.header_cards())
    header_cards.update(ds.calibrator.header_cards())
    
    for key, value in header_cards.items():
        primary_hdu.header[key] = value
    add_comments('primary', primary_hdu.header)
    
    return primary_hdu

def construct_psrparam_hdu(ds):
    '''
    Construct PSRPARAM HDU
    '''
    param_data = np.array(
        [line for line in ds.source.model.split('\n')],
        dtype=(np.record, [('PARAM', 'S128')])
    )
    psrparam_hdu = fits.BinTableHDU(data=param_data)
    psrparam_hdu.header['extname'] = 'PSRPARAM'
    psrparam_hdu.header['extver'] = 1
    add_comments('psrparam', psrparam_hdu.header)
    
    return psrparam_hdu

def construct_t2predict_hdu(ds):
    '''
    Construct Tempo2 predictor HDU
    '''
    predictor_data = np.array(
        [line for line in ds.source.model.split('\n')],
        dtype=(np.record, [('PREDICT', 'S128')])
    )
    t2predict_hdu = fits.BinTableHDU(data=predictor_data)
    t2predict_hdu.header['extname'] = 'T2PREDICT'
    t2predict_hdu.header['extver'] = 1
    add_comments('t2predict', t2predict_hdu.header)
    
    return t2predict_hdu

def construct_polyco_hdu(ds):
    '''
    Construct Polyco HDU
    '''
    polyco_data = ds.source.polyco.as_table()
    polyco_hdu = fits.BinTableHDU(data=polyco_data)
    polyco_hdu.header['tunit7'] = 'MHz'
    polyco_hdu.header['tunit11'] = 'Hz'
    polyco_hdu.header['extname'] = 'POLYCO'
    polyco_hdu.header['extver'] = 1
    add_comments('polyco', polyco_hdu.header)
    
    return polyco_hdu

def construct_history_hdu(ds):
    '''
    Construct History HDU
    '''
    table = ds.history.as_table()
    history_hdu = fits.BinTableHDU(data=table)
    history_hdu.header['tunit9'] = 's'
    history_hdu.header['tunit10'] = 'MHz'
    history_hdu.header['tunit12'] = 'MHz'
    history_hdu.header['tunit13'] = 'pc.cm-3'
    history_hdu.header['tunit14'] = 'rad'
    history_hdu.header['extname'] = 'HISTORY'
    history_hdu.header['extver'] = 1
    add_comments('history', history_hdu.header)
    
    return history_hdu

def construct_subint_hdu(ds):
    '''
    Construct Subintegration HDU
    '''
    if hasattr(ds, "ra") and hasattr(ds, "dec"):
        ra = ds.ra
        dec = ds.dec
    else:
        coords = ds.observation.coords
        ra_val = coords.ra.to(u.deg).value
        dec_val = coords.dec.to(u.deg).value
        ra = np.full_like(ds.time, ra_val)
        dec = np.full_like(ds.time, dec_val)
    
    if hasattr(ds, "glon") and hasattr(ds, "glat"):
        glon = ds.glon
        glat = ds.glat
    else:
        coords = ds.observation.coords
        l_val = coords.transform_to(Galactic).l.to(u.deg).value
        b_val = coords.transform_to(Galactic).b.to(u.deg).value
        glon = np.full_like(ds.time, l_val)
        glat = np.full_like(ds.time, b_val)
    
    data = np.array([getattr(ds, pol) for pol in get_pols(ds)])
    data = np.swapaxes(data, 0, 1)
    mins = np.min(data, axis=-1)
    maxes = np.max(data, axis=-1)
    mant, expt = np.frexp(maxes - mins)
    offsets = (mins + maxes)/2.
    scales = 2.**(expt-16)
    data -= offsets[..., np.newaxis]
    data /= scales[..., np.newaxis]
    data = np.rint(data).astype('i2')
    
    coords = ds.observation.coords
    subint_data = np.rec.fromarrays(
        [
            np.arange(ds.time.size),
            ds.duration.data,
            ds.time.data,
            ds.lst.data,
            ra,
            dec,
            glon,
            glat,
            ds.feed_angle if hasattr(ds, "feed_angle") else np.zeros_like(ds.time),
            ds.pos_angle if hasattr(ds, "pos_angle") else np.zeros_like(ds.time),
            ds.par_angle if hasattr(ds, "par_angle") else np.zeros_like(ds.time),
            ds.az.data,
            ds.zen.data,
            ds.aux_dm if hasattr(ds, "aux_dm") else np.zeros_like(ds.time),
            ds.aux_rm if hasattr(ds, "aux_rm") else np.zeros_like(ds.time),
            np.tile(ds.freq, ds.time.size).reshape(ds.time.size, -1),
            ds.weights,
            offsets.reshape(ds.time.size, -1),
            scales.reshape(ds.time.size, -1),
            data,
        ],
        dtype=(np.record, [
                ('INDEXVAL', '>f8'),
                ('TSUBINT', '>f8'),
                ('OFFS_SUB', '>f8'),
                ('LST_SUB', '>f8'),
                ('RA_SUB', '>f8'),
                ('DEC_SUB', '>f8'),
                ('GLON_SUB', '>f8'),
                ('GLAT_SUB', '>f8'),
                ('FD_ANG', '>f4'),
                ('POS_ANG', '>f4'),
                ('PAR_ANG', '>f4'),
                ('TEL_AZ', '>f4'),
                ('TEL_ZEN', '>f4'),
                ('AUX_DM', '>f8'),
                ('AUX_RM', '>f8'),
                ('DAT_FREQ', '>f8', (ds.freq.size,)),
                ('DAT_WTS', '>f4', (ds.freq.size,)),
                ('DAT_OFFS', '>f4', (ds.n_polns*ds.freq.size,)),
                ('DAT_SCL', '>f4', (ds.n_polns*ds.freq.size,)),
                ('DATA', '>i2', (ds.n_polns, ds.freq.size, ds.phase.size)),
            ]),
        )
    
    subint_hdu = fits.BinTableHDU(data=subint_data)
    
    header_cards = {
        'int_type': ds.time_var,
        'int_unit': ds.time_unit,
        'scale': ds.flux_unit,
        'pol_type': ds.pol_type,
        'npol': ds.n_polns,
        'tbin': ds.time_per_bin,
        'nbin': ds.phase.shape[0],
        'nbin_prd': int(1/(ds.phase.data[1]-ds.phase.data[0])),
        'phs_offs': ds.phase.data[0],
        'nbits': 1, # search mode data not currently supported
        'zero_off': '*',
        'signint': 0,
        'nsuboffs': '*',
        'nchan': ds.freq.shape[0],
        'chan_bw': ds.channel_bandwidth,
        'DM': ds.DM,
        'RM': ds.RM,
        'nchnoffs': if_missing('*', ds.channel_offset),
        'nsblk': 1,
        'nstot': '*',
        'epochs': ds.epoch_type,
        'tunit2': 's',
        'tunit3': 's',
        'tunit4': 's',
        'tunit5': 'deg',
        'tunit6': 'deg',
        'tunit7': 'deg',
        'tunit8': 'deg',
        'tunit9': 'deg',
        'tunit10': 'deg',
        'tunit11': 'deg',
        'tunit12': 'deg',
        'tunit13': 'deg',
        'tunit14': 'pc cm-3',
        'tunit15': 'rad m-2',
        'tunit16': 'MHz',
        'tunit20': 'Jy',
        'extname': 'SUBINT',
        'extver': 1,
    }
    
    for key, value in header_cards.items():
        subint_hdu.header[key] = value
    add_comments('subint', subint_hdu.header)
    
    return subint_hdu

def add_comments(hdu_name, header):
    base_dir = os.path.dirname(__file__)
    comments_file = os.path.join(base_dir, "standard-comments.toml")
    comments = toml.load(comments_file)
    for key, value in comments[hdu_name].items():
        header.comments[key] = value
    
