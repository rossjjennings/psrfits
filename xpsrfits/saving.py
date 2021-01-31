import numpy as np
import xarray as xr
import warnings
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from datetime import datetime
from textwrap import dedent

def to_hdulist(ds):
    '''
    Convert an xpsrfits dataset to a FITS HDU list for saving.
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
    smjd = 86400*(ds.start_time.mjd_long - imjd) # not leap second accurate?
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
        'stt_lst': '*',
    }
    
    header_cards.update(ds.source.header_cards())
    header_cards.update(ds.observation.header_cards())
    header_cards.update(ds.telescope.header_cards())
    header_cards.update(ds.frontend.header_cards())
    header_cards.update(ds.backend.header_cards())
    header_cards.update(ds.beam.header_cards())
    header_cards.update(ds.calibrator.header_cards())
    
    for key in header_cards:
        primary_hdu.header[key] = header_cards[key]
    
    return fits.HDUList([primary_hdu])
