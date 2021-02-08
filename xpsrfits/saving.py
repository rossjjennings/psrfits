import numpy as np
import xarray as xr
import warnings
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from datetime import datetime
from textwrap import dedent
import toml
import os.path

from .attrs.attrcollection import if_missing

def to_hdulist(ds):
    '''
    Convert an xpsrfits dataset to a FITS HDU list for saving.
    '''
    base_dir = os.path.dirname(__file__)
    comments_file = os.path.join(base_dir, "standard-comments.toml")
    comments = toml.load(comments_file)
    hdus = []
    
    # Construct primary HDU
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
    
    for key, value in comments['primary'].items():
        primary_hdu.header.comments[key] = value
    
    hdus.append(primary_hdu)
    
    # Construct PSRPARAM HDU
    param_data = np.array(
        [line for line in ds.source.model.split('\n')],
        dtype=(np.record, [('PARAM', 'S128')])
    )
    psrparam_hdu = fits.BinTableHDU(data=param_data)
    psrparam_hdu.header['extname'] = 'PSRPARAM'
    psrparam_hdu.header['extver'] = 1
    
    for key, value in comments['psrparam'].items():
        psrparam_hdu.header.comments[key] = value
    
    hdus.append(psrparam_hdu)
    
    # Construct Tempo2 predictor HDU, if appropriate
    if ds.source.predictor is not None:
        predictor_data = np.array(
            [line for line in ds.source.model.split('\n')],
            dtype=(np.record, [('PREDICT', 'S128')])
        )
        t2predict_hdu = fits.BinTableHDU(data=predictor_data)
        t2predict_hdu.header['extname'] = 'T2PREDICT'
        t2predict_hdu.header['extver'] = 1
        
        for key, value in comments['t2predict'].items():
            psrparam_hdu.header.comments[key] = value
        
        hdus.append(t2predict_hdu)
    
    # Construct Polyco HDU, if appropriate
    if ds.source.polyco is not None:
        polyco_data = ds.source.polyco.as_table()
        polyco_hdu = fits.BinTableHDU(data=polyco_data)
        polyco_hdu.header['tunit7'] = 'MHz'
        polyco_hdu.header['tunit11'] = 'Hz'
        polyco_hdu.header['extname'] = 'POLYCO'
        polyco_hdu.header['extver'] = 1
        
        for key, value in comments['polyco'].items():
            polyco_hdu.header.comments[key] = value
        
        hdus.append(polyco_hdu)
    
    # Construct History HDU
    table = ds.history.as_table()
    history_hdu = fits.BinTableHDU(data=table)
    history_hdu.header['tunit9'] = 's'
    history_hdu.header['tunit10'] = 'MHz'
    history_hdu.header['tunit12'] = 'MHz'
    history_hdu.header['tunit13'] = 'pc.cm-3'
    history_hdu.header['tunit14'] = 'rad'
    history_hdu.header['extname'] = 'HISTORY'
    history_hdu.header['extver'] = 1
    
    for key, value in comments['history'].items():
        history_hdu.header.comments[key] = value
    
    hdus.append(history_hdu)
    
    # Construct Subintegration HDU
    subint_hdu = fits.BinTableHDU()
    subint_header_cards = {
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
    
    for key, value in subint_header_cards.items():
        subint_hdu.header[key] = value
    
    for key, value in comments['subint'].items():
        if key in subint_header_cards:
            subint_hdu.header.comments[key] = value
    
    hdus.append(subint_hdu)

    
    return fits.HDUList(hdus)
