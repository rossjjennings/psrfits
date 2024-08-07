import numpy as np
from textwrap import indent, dedent
from datetime import datetime
from astropy.time import Time
import astropy.units as u
import copy

from .attrcollection import AttrCollection, maybe_missing, if_missing

class History:
    @classmethod
    def from_hdu(cls, hdu):
        out = cls()
        out.entries = HistoryEntry.from_table(hdu.data)
        return out
    
    def __str__(self):
        return f"<History with {len(self.entries)} entries>"
    
    def __repr__(self):
        description = "<psrfits.History>\nLatest entry:\n"
        description += indent(self.entries[-1]._repr_items(), '    ')
        return description
    
    def __getattr__(self, name):
        return getattr(self.entries[-1], name)
    
    def __getitem__(self, key):
        return self.entries[key]
    
    def __len__(self):
        return len(self.entries)
    
    def __iter__(self):
        for entry in self.entries:
            yield entry

    def add_entry(self, **kwargs):
        new_entry = copy.copy(self.entries[-1])
        new_entry.date = Time.now()
        for key in kwargs:
            setattr(new_entry, key, kwargs[key])
        self.entries.append(new_entry)

    def as_table(self):
        return np.array(
            [(
                datetime.strftime(entry.date.datetime, '%a %b %d %H:%M:%S %Y'),
                if_missing('UNKNOWN', entry.command),
                entry.flux_unit,
                entry.pol_type,
                entry.n_subints,
                entry.n_polns,
                entry.n_bins,
                entry.bins_per_period,
                entry.time_per_bin.to(u.s).value,
                entry.center_freq.to(u.MHz).value,
                entry.n_channels,
                entry.channel_bandwidth.to(u.MHz).value,
                entry.DM.to(u.pc/u.cm**3).value,
                entry.RM.to(u.rad/u.m**2).value,
                int(entry.projection_corrected),
                int(entry.feed_corrected),
                int(entry.backend_corrected),
                int(entry.rm_corrected),
                int(entry.dedispersed),
                if_missing('UNSET', entry.dedisp_method),
                if_missing('NONE', entry.scatter_method),
                if_missing('NONE', entry.cal_method),
                if_missing('NONE', entry.cal_file),
                if_missing('NONE', entry.rfi_method),
                if_missing('NONE', entry.rm_model),
                int(entry.aux_rm_corrected),
                if_missing('NONE', entry.dm_model),
                int(entry.aux_dm_corrected),
            ) for entry in self],
            dtype = (np.record, [
                ('DATE_PRO', 'S24'),
                ('PROC_CMD', 'S256'),
                ('SCALE', 'S8'),
                ('POL_TYPE', 'S8'),
                ('NSUB', '>i4'),
                ('NPOL', '>i2'),
                ('NBIN', '>i2'),
                ('NBIN_PRD', '>i2'),
                ('TBIN', '>f8'),
                ('CTR_FREQ', '>f8'),
                ('NCHAN', '>i4'),
                ('CHAN_BW', '>f8'),
                ('DM', '>f8'),
                ('RM', '>f8'),
                ('PR_CORR', '>i2'),
                ('FD_CORR', '>i2'),
                ('BE_CORR', '>i2'),
                ('RM_CORR', '>i2'),
                ('DEDISP', '>i2'),
                ('DDS_MTHD', 'S32'),
                ('SC_MTHD', 'S32'),
                ('CAL_MTHD', 'S32'),
                ('CAL_FILE', 'S256'),
                ('RFI_MTHD', 'S32'),
                ('RM_MODEL', 'S32'),
                ('AUX_RM_C', '>i2'),
                ('DM_MODEL', 'S32'),
                ('AUX_DM_C', '>i2'),
            ]),
        )

class HistoryEntry(AttrCollection):
    __slots__ = (
        'date',
        'command',
        'flux_unit',
        'pol_type',
        'n_subints',
        'n_polns',
        'n_bins',
        'bins_per_period',
        'time_per_bin',
        'center_freq',
        'n_channels',
        'channel_bandwidth',
        'DM',
        'RM',
        'projection_corrected',
        'feed_corrected',
        'backend_corrected',
        'rm_corrected',
        'dedispersed',
        'dedisp_method',
        'scatter_method',
        'cal_method',
        'cal_file',
        'rfi_method',
        'rm_model',
        'aux_rm_corrected',
        'dm_model',
        'aux_dm_corrected',
    )
    
    @classmethod
    def from_table(cls, table):
        entries = [{} for i in range(table.size)]
        for i in range(table.size):
            try:
                timestamp = datetime.strptime(table['date_pro'][i], '%a %b %d %H:%M:%S %Y')
            except ValueError:
                timestamp = datetime.strptime(table['date_pro'][i], '%Y-%m-%d %H:%M:%S.%f')
            entries[i] = {
                'date': Time(timestamp),
                'command': maybe_missing(table['proc_cmd'][i]),
                'flux_unit': table['scale'][i],
                'pol_type': table['pol_type'][i],
                'n_subints': table['nsub'][i],
                'n_polns': table['npol'][i],
                'n_bins':table['nbin'][i],
                'bins_per_period': table['nbin_prd'][i],
                'time_per_bin': table['tbin'][i]*u.s,
                'center_freq': table['ctr_freq'][i]*u.MHz,
                'n_channels': table['nchan'][i],
                'channel_bandwidth': table['chan_bw'][i]*u.MHz,
                'DM': table['DM'][i]*u.pc/u.cm**3,
                'RM': table['RM'][i]*u.rad/u.m**2,
                'projection_corrected': bool(table['pr_corr'][i]),
                'feed_corrected': bool(table['fd_corr'][i]),
                'backend_corrected': bool(table['be_corr'][i]),
                'rm_corrected': bool(table['rm_corr'][i]),
                'dedispersed': bool(table['dedisp'][i]),
                'dedisp_method': maybe_missing(table['dds_mthd'][i]),
                'scatter_method': maybe_missing(table['sc_mthd'][i]),
                'cal_method': maybe_missing(table['cal_mthd'][i]),
                'cal_file': maybe_missing(table['cal_file'][i]),
                'rfi_method': maybe_missing(table['rfi_mthd'][i]),
                'rm_model': maybe_missing(table['rm_model'][i]),
                'aux_rm_corrected': bool(table['aux_rm_c'][i]),
                'dm_model': maybe_missing(table['dm_model'][i]),
                'aux_dm_corrected': bool(table['aux_dm_c'][i]),
            }
        return [cls(**entry) for entry in entries]
    
    def __str__(self):
        return f"<HistoryEntry from {self.date}>"
    
    def __repr__(self):
        description = "<psrfits.HistoryEntry>\n"
        description += indent(self._repr_items(), '    ')
        return description
