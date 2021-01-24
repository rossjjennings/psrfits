from textwrap import indent, dedent
from datetime import datetime
from astropy.time import Time

from .attrcollection import AttrCollection, maybe_missing

class History:
    def __init__(self, entries):
        self.entries = HistoryEntry.from_table(entries)
    
    @classmethod
    def from_hdu(cls, hdu):
        return cls(hdu.data)
    
    def __str__(self):
        return f"<History with {len(self.entries)} entries>"
    
    def __repr__(self):
        description = "<xpsrfits.History>\nLatest entry:\n"
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
            timestamp = datetime.strptime(table['date_pro'][i], '%a %b %d %H:%M:%S %Y')
            entries[i]['date'] = Time(timestamp)
            entries[i]['command'] = maybe_missing(table['proc_cmd'][i]) # UNKNOWN
            entries[i]['flux_unit'] = table['scale'][i]
            entries[i]['pol_type'] = table['pol_type'][i]
            entries[i]['n_subints'] = table['nsub'][i]
            entries[i]['n_polns'] = table['npol'][i]
            entries[i]['n_bins'] = table['nbin'][i]
            entries[i]['bins_per_period'] = table['nbin_prd'][i]
            entries[i]['time_per_bin'] = table['tbin'][i]
            entries[i]['center_freq'] = table['ctr_freq'][i]
            entries[i]['n_channels'] = table['nchan'][i]
            entries[i]['channel_bandwidth'] = table['chan_bw'][i]
            entries[i]['DM'] = table['DM'][i]
            entries[i]['RM'] = table['RM'][i]
            entries[i]['projection_corrected'] = bool(table['pr_corr'][i])
            entries[i]['feed_corrected'] = bool(table['fd_corr'][i])
            entries[i]['backend_corrected'] = bool(table['be_corr'][i])
            entries[i]['rm_corrected'] = bool(table['rm_corr'][i])
            entries[i]['dedispersed'] = bool(table['dedisp'][i])
            entries[i]['dedisp_method'] = maybe_missing(table['dds_mthd'][i]) # UNSET
            entries[i]['scatter_method'] = maybe_missing(table['sc_mthd'][i]) # NONE
            entries[i]['cal_method'] = maybe_missing(table['cal_mthd'][i]) # NONE
            entries[i]['cal_file'] = maybe_missing(table['cal_file'][i]) # NONE
            entries[i]['rfi_method'] = maybe_missing(table['rfi_mthd'][i]) # NONE
            entries[i]['rm_model'] = maybe_missing(table['rm_model'][i]) # NONE
            entries[i]['aux_rm_corrected'] = bool(table['aux_rm_c'][i])
            entries[i]['dm_model'] = maybe_missing(table['dm_model'][i]) # NONE
            entries[i]['aux_dm_corrected'] = bool(table['aux_dm_c'][i])
        return [cls(**entry) for entry in entries]
    
    def __str__(self):
        return f"<HistoryEntry from {self.date}>"
    
    def __repr__(self):
        description = "<xpsrfits.HistoryEntry>\n"
        description += indent(self._repr_items(), '    ')
        return description
