import numpy as np
from astropy.time import Time
from psrfits.formatting import fmt_items, fmt_array
from psrfits import baseline, dispersion, polarization, plots, averaging

class Dataset:
    def __init__(self, **attrs):
        for attr, val in attrs.items():
            setattr(self, attr, val)

    def copy(self):
        '''
        Make a copy of this Dataset. Array-like attributes, including Numpy arrays
        and Astropy Quantity and Time objects, are copied, but other attributes,
        including Dask arrays, which are meant to be immutable, are not.
        '''
        attrs = {}
        for attr, value in self.__dict__.items():
            if isinstance(attr, np.ndarray) or isinstance(attr, Time):
                attrs[attr] = value.copy()
            else:
                attrs[attr] = value
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
        Print a summary of the data contained in this Dataset.
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
        Print a formatted list of all attributes of this Dataset.
        '''
        print(fmt_items(self.__dict__))

    def dedisperse(self, DM=None, weight_center_freq=False):
        '''
        Dedisperse the data with the given DM.
        If `DM` is `None`, use the DM attribute of `ds`.
        '''
        return dispersion.dedisperse(self, inplace=True, DM=DM, weight_center_freq=weight_center_freq)

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
