import numpy as np
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

def save(filename, ds, overwrite=False):
    hdul = to_hdulist(ds)
    hdul.writeto(filename, overwrite=overwrite)

def to_hdulist(ds):
    '''
    Convert an psrfits dataset to a FITS HDU list for saving.
    '''
    hdus = []
    hdus.append(construct_primary_hdu(ds))
    hdus.append(construct_history_hdu(ds))
    if hasattr(ds, 'model'):
        hdus.append(construct_psrparam_hdu(ds))
    if hasattr(ds, 'predictor'):
        hdus.append(construct_t2predict_hdu(ds))
    if hasattr(ds, 'polyco'):
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
        # Setting HDRVER = '6.1' makes PSRCHIVE look for (and not find)
        # a column named 'REF_FREQ' in the generated file.
        # So, leaving as '5.4' for now.
        'hdrver': '5.4',
        'date': datetime.strftime(datetime.now(), '%Y-%m-%dT%H:%M:%S'),
        'obsfreq': ds.center_freq.to(u.MHz).value,
        'obsbw': ds.bandwidth.to(u.MHz).value,
        'obsnchan': ds.history[0].n_channels,
        'chan_dm': ds.DM.to(u.pc/u.cm**3).value,
        'src_name': ds.source,
        'stt_imjd': imjd,
        'stt_smjd': int(smjd),
        'stt_offs': smjd % 1,
        'stt_lst': 3600*ds.start_lst.to(u.hourangle).value,
    }
    
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
        [line for line in ds.model.split('\n')],
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
    description = ds.predictor.describe()
    predictor_data = np.array(
        [line for line in description.split('\n')],
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
    polyco_data = ds.polyco.as_table()
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
    history_hdu.header['tunit13'] = 'pc cm-3'
    history_hdu.header['tunit14'] = 'rad'
    history_hdu.header['extname'] = 'HISTORY'
    history_hdu.header['extver'] = 1
    add_comments('history', history_hdu.header)
    
    return history_hdu

def construct_subint_hdu(ds):
    '''
    Construct Subintegration HDU
    '''
    data = np.stack([getattr(ds, pol) for pol in get_pols(ds)])
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

    items = []
    items.append((('INDEXVAL', '>f8'), ds.index))
    items.append((('TSUBINT', '>f8'), ds.duration.data))
    items.append((('OFFS_SUB', '>f8'), (ds.epoch - ds.start_time).to(u.s).value))
    if hasattr(ds, 'lst'):
        items.append((('LST_SUB', '>f8'), ds.lst.hourangle*3600))
    if hasattr(ds, 'coords'):
        items.append((('RA_SUB', '>f8'), ds.coords.ra.deg))
        items.append((('DEC_SUB', '>f8'), ds.coords.dec.deg))
    if hasattr(ds, 'coords_galactic'):
        items.append((('GLON_SUB', '>f8'), ds.coords_galactic.l.deg))
        items.append((('GLAT_SUB', '>f8'), ds.coords_galactic.b.deg))
    if hasattr(ds, 'feed_angle'):
        items.append((('FD_ANG', '>f4'), ds.feed_angle))
    if hasattr(ds, 'pos_angle'):
        items.append((('POS_ANG', '>f4'), ds.pos_angle))
    if hasattr(ds, 'par_angle'):
        items.append((('PAR_ANG', '>f4'), ds.par_angle))
    if hasattr(ds, 'coords_altaz'):
        items.append((('TEL_AZ', '>f4'), ds.coords_altaz.az.deg))
        items.append((('TEL_ZEN', '>f4'), 90 - ds.coords_altaz.alt.deg))
    items.append((('AUX_DM', '>f8'), ds.aux_dm.to(u.pc/u.cm**3).value))
    items.append((('AUX_RM', '>f8'), ds.aux_rm.to(u.rad/u.m**2).value))
    items.append((('DAT_FREQ', '>f8', (ds.freq.size,)),
                np.tile(ds.freq, ds.epoch.size).reshape(ds.epoch.size, -1)))
    items.append((('DAT_WTS', '>f4', (ds.freq.size,)), ds.weights))
    items.append((('DAT_OFFS', '>f4', (ds.n_polns*ds.freq.size,)),
                 offsets.reshape(ds.epoch.size, -1)))
    items.append((('DAT_SCL', '>f4', (ds.n_polns*ds.freq.size,)),
                 scales.reshape(ds.epoch.size, -1)))
    items.append((('DATA', '>i2', (ds.n_polns, ds.freq.size, ds.phase.size)), data))

    subint_data = np.rec.fromarrays(
        [item[1] for item in items],
        dtype=(np.record, [item[0] for item in items])
    )
    
    subint_hdu = fits.BinTableHDU(data=subint_data)
    
    header_cards = {
        'int_type': ds.time_var,
        'int_unit': ds.time_unit,
        'scale': ds.flux_unit,
        'pol_type': ds.pol_type,
        'npol': ds.n_polns,
        'tbin': ds.history.time_per_bin.to(u.s).value,
        'nbin': ds.phase.shape[0],
        'nbin_prd': int(1/(ds.phase.data[1]-ds.phase.data[0])),
        'phs_offs': ds.phase.data[0],
        'nbits': 1, # search mode data not currently supported
        'zero_off': '*',
        'signint': 0,
        'nsuboffs': '*',
        'nchan': ds.freq.shape[0],
        'chan_bw': ds.channel_bandwidth.to(u.MHz).value,
        'DM': ds.DM.to(u.pc/u.cm**3).value,
        'RM': ds.RM.to(u.rad/u.m**2).value,
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
    
