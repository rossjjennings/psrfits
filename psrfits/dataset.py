import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord
import copy

from psrfits.formatting import fmt_items, fmt_array
from psrfits import baseline, dispersion, polarization, plots, averaging
from psrfits.attrs.history import History

class Dataset:
    '''
    A set of data that is homogeneous (same source, receiver, frequency range, etc.),
    but may contain data from multiple PSRFITS files.
    '''
    @classmethod
    def collect(cls, datafiles):
        '''
        Collect data from several DataFile objects into a Dataset.
        '''
        datafiles.sort(key=lambda item: item.start_time)
        first = datafiles[0]
        ref_freq = first.freq

        out = cls()
        out.which_file = np.concatenate(
            [np.full(df.epoch.shape, i) for i, df in enumerate(datafiles)]
        )

        # always use the first value for start_time and start_lst
        out.start_time = first.start_time
        out.start_lst = first.start_lst

        # other coordinates
        assert all(np.all(df.freq == first.freq) for df in datafiles)
        out.freq = first.freq
        assert all(np.all(df.phase == first.phase) for df in datafiles)
        out.phase = first.phase

        # these should match for homogeneous data
        assert all(df.center_freq == first.center_freq for df in datafiles)
        out.center_freq = first.center_freq
        assert all(df.bandwidth == first.bandwidth for df in datafiles)
        out.bandwidth = first.bandwidth
        assert all(df.channel_bandwidth == first.channel_bandwidth for df in datafiles)
        out.channel_bandwidth = first.channel_bandwidth
        assert all(df.n_polns == first.n_polns for df in datafiles)
        out.n_polns = first.n_polns
        assert all(df.pol_type == first.pol_type for df in datafiles)
        out.pol_type = first.pol_type
        assert all(df.epoch_type == first.epoch_type for df in datafiles)
        out.epoch_type = first.epoch_type
        assert all(df.time_var == first.time_var for df in datafiles)
        out.time_var = first.time_var
        assert all(df.time_unit == first.time_unit for df in datafiles)
        out.time_unit = first.time_unit

        # deal with channel_offset?

        # TODO: Use aux_dm and aux_rm here?
        assert all(df.DM == first.DM for df in datafiles)
        out.DM = first.DM
        assert all(df.RM == first.RM for df in datafiles)
        out.RM = first.RM

        # time series attributes can just be concatenated
        out.epoch = Time(np.concatenate([df.epoch for df in datafiles]))
        out.duration = np.concatenate([df.duration for df in datafiles])
        out.index = np.concatenate([df.index for df in datafiles])
        if all(hasattr(df, 'lst') for df in datafiles):
            out.lst = np.concatenate([df.lst for df in datafiles])
        if all(hasattr(df, 'coords') for df in datafiles):
            out.coords = SkyCoord(np.concatenate([df.coords for df in datafiles]))
        if all(hasattr(df, 'feed_angle') for df in datafiles):
            out.feed_angle = np.concatenate([df.feed_angle for df in datafiles])
        if all(hasattr(df, 'pos_angle') for df in datafiles):
            out.pos_angle = np.concatenate([df.pos_angle for df in datafiles])
        if all(hasattr(df, 'par_angle') for df in datafiles):
            out.par_angle = np.concatenate([df.par_angle for df in datafiles])
        if all(hasattr(df, 'coords_galactic') for df in datafiles):
            out.coords_galactic = SkyCoord(np.concatenate([df.coords_galactic for df in datafiles]))
        if all(hasattr(df, 'coords_altaz') for df in datafiles):
            out.coords_altaz = SkyCoord(np.concatenate([df.coords for df in datafiles]))
        out.aux_dm = np.concatenate([df.aux_dm for df in datafiles])
        out.aux_rm = np.concatenate([df.aux_rm for df in datafiles])

        out.weights = np.concatenate([df.weights for df in datafiles])
        out.frequencies = np.concatenate([df.frequencies for df in datafiles])
        for pol in first.get_pols():
            setattr(out, pol, np.concatenate([getattr(df, pol) for df in datafiles]))

        return out

    def copy(self):
        '''
        Make a copy of this Dataset. Array-like attributes, including Numpy arrays
        and Astropy Quantity and Time objects, are copied, but other attributes,
        including Dask arrays, which are meant to be immutable, are not.
        '''
        out = self.__class__()
        for attr, value in self.__dict__.items():
            if isinstance(value, np.ndarray) or isinstance(value, Time):
                setattr(out, attr, value.copy())
            elif isinstance(value, History):
                history = History()
                history.entries = copy.deepcopy(self.history.entries)
                setattr(out, attr, history)
            else:
                setattr(out, attr, value)
        return out

    def __repr__(self):
        return (
            f"<psrfits.{self.__class__.__name__}: "
            f"{self.observation.mode} {self.source}, "
            f"{self.frontend.name} {self.backend.name} ({self.telescope.name}), "
            f"{self.start_time.iso}"
            ">"
        )

    def info(self):
        '''
        Print a summary of the data contained in this Dataset.
        '''
        pols = self.get_pols()
        npol = len(pols)
        nsub, nchan, nbin = getattr(self, pols[0]).shape
        info_items = {
            'Source': self.source,
            'Mode': self.observation.mode,
            'Telescope': self.telescope.name,
            'Frontend': self.frontend.name,
            'Backend': self.backend.name,
            'Project ID': self.observation.project_id,
            'Start Time': self.start_time,
            'Duration': f"{np.sum(self.duration):g}",
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
        Print a formatted list of all attributes of this Dataset.
        '''
        print(fmt_items(self.__dict__))

    def dedisperse(self, DM=None, ref_freq=None):
        '''
        Dedisperse the data with the given DM and reference frequency.
        If `DM` is `None`, use the DM attribute of `ds`.
        If `ref_freq` is `None`, use the polyco reference frequency.
        '''
        return dispersion.dedisperse(self, inplace=True, DM=DM, ref_freq=ref_freq)

    def align_with_predictor(self, out_of_bounds='error'):
        '''
        Dedisperse and align the data using the internal Tempo2 predictor.
        '''
        return dispersion.align_with_predictor(self, inplace=True, out_of_bounds=out_of_bounds)

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
        return baseline.remove_baseline(self, inplace=True, method=method, frac=frac)

    def get_pols(self):
        '''
        Return a list of the polarizations present in the data.
        '''
        return polarization.get_pols(self)

    def pscrunch(self):
        '''
        Return a dataset containing only the total intensity component of the input.
        '''
        return polarization.pscrunch(self, inplace=True)

    def to_stokes(self):
        '''
        Transform coherence (AABBCRCI) data to Stokes parameters (IQUV).
        If input is already Stokes, leave it alone.
        If input has one or two polarizations, return I only.
        '''
        return polarization.to_stokes(self, inplace=True)

    def avg_profile(self, poln='I', low_freq=-np.inf, high_freq=np.inf, start_mjd=-np.inf,
                    stop_mjd=np.inf, use_weights=True, compute=True, progress=False):
        '''
        Calculate an average profile from the data.

        Parameters
        ----------
        poln: Polarization to use
        low_freq: Low frequency, as a Quantity. Lower-frequency data will be excluded.
        high_freq: High frequency, as a Quantity. Higher-frequency data will be excluded.
        start_mjd: Initial time, as an MJD. Data before this point will be excluded.
        stop_mjd: Final time, as an MJD. Data after this point will be excluded.
        use_weights: Compute a weighted average. If `True`, the weights stored in the
                    Dataset will be used. If `False`, no weights will be used.
        compute: Call compute() on the result, if it is a Dask array.
        progress: Display a progress bar during the computation.
        '''
        return averaging.avg_profile(self, poln, low_freq, high_freq, start_mjd, stop_mjd, use_weights)

    def avg_portrait(self, poln='I', start_mjd=-np.inf, stop_mjd=np.inf, use_weights=True,
                     compute=True, progress=False):
        '''
        Calculate an average portrait (frequency vs. pulse phase) from the data.

        Parameters
        ----------
        poln: Polarization to use
        start_mjd: Initial time, as an MJD. Data before this point will be excluded.
        stop_mjd: Final time, as an MJD. Data after this point will be excluded.
        use_weights: Compute a weighted average. If `True`, the weights stored in the
                    Dataset will be used. If `False`, no weights will be used.
        compute: Call compute() on the result, if it is a Dask array.
        progress: Display a progress bar during the computation.
        '''
        return averaging.avg_portrait(self, poln, start_mjd, stop_mjd, use_weights)

    def avg_pulsetrain(self, poln='I', low_freq=-np.inf, high_freq=np.inf, use_weights=True,
                       compute=True, progress=False):
        '''
        Calculate an average pulse train (time vs. pulse phase, a bit of a misnomer)
        from the data.

        Parameters
        ----------
        poln: Polarization to use
        low_freq: Low frequency, as a Quantity. Lower-frequency data will be excluded.
        high_freq: High frequency, as a Quantity. Higher-frequency data will be excluded.
        use_weights: Compute a weighted average. If `True`, the weights stored in the
                    Dataset will be used. If `False`, no weights will be used.
        compute: Call compute() on the result, if it is a Dask array.
        progress: Display a progress bar during the computation.
        '''
        return averaging.avg_pulsetrain(self, poln, low_freq, high_freq, use_weights)

    def plot_profile(self, profile, ax=None, **kwargs):
        '''
        Make a line plot of a supplied pulse profile using metadata from this
        Dataset. Additional keyword arguments will be passed on to plt.plot().

        Parameters
        ----------
        profile: Profile (array of data vs. pulse phase) to plot
        ax:      Axes in which to plot. If `None`, the current Axes will be used.
        '''
        return plots.plot_profile(self, profile, ax, **kwargs)

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
        return plots.plot_portrait(self, profile, ax, sym_lim, vmin, vmax, **kwargs)

    def plot_pulsetrain(self, pulsetrain, ax=None, sym_lim=False, vmin=None, vmax=None, **kwargs):
        '''
        Make a pseudocolor plot of a supplied "pulse train" (i.e., time series of profiles
        matching the length of the underlying data, a bit of a misnomer) using metadata
        from this Dataset. Additional keyword arguments will be passed on to plt.pcolormesh().

        Parameters
        ----------
        profile: Profile (array of data vs. pulse phase) to plot
        ax:      Axes in which to plot. If `None`, the current Axes will be used.
        sym_lim: Symmetrize the colorbar limits around zero. Useful when plotting
                 signed data using a diverging colormap.
        '''
        return plots.plot_pulsetrain(self, pulsetrain, ax, sym_lim, vmin, vmax, **kwargs)

    def plot_freqtime(self, data, ax=None, sym_lim=False, vmin=None, vmax=None, **kwargs):
        '''
        Make a pseudocolor plot of a set of data as a function of frequency and time
        using metadata from the given Dataset. Additional keyword arguments will be passed on to plt.pcolormesh().

        Parameters
        ----------
        profile: Profile (array of data vs. pulse phase) to plot
        ax:      Axes in which to plot. If `None`, the current Axes will be used.
        sym_lim: Symmetrize the colorbar limits around zero. Useful when plotting
                signed data using a diverging colormap.
        '''
        return plots.plot_freqtime(self, data, ax, sym_lim, vmin, vmax, **kwargs)
