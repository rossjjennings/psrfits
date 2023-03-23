import numpy as np
from numpy.polynomial import chebyshev
from astropy.time import Time
import astropy.units as u
from pint import PulsarMJD

class ChebyModel:
    """
    A model of pulse phase as a function of frequency and time defined by a
    2-D Chebyshev Polynomial combined with a dispersion constant, as used by Tempo2.
    """
    def __init__(self, psrname, sitename, start_time, end_time, start_freq, end_freq,
                 dispersion_constant, coeffs):
        self.psrname = psrname
        self.sitename = sitename
        self.start_time = start_time
        self.end_time = end_time
        self.start_freq = start_freq
        self.end_freq = end_freq
        self.dispersion_constant = dispersion_constant
        self.coeffs = coeffs

    @classmethod
    def parse(cls, lines):
        """
        Construct a ChebyModel object by reading lines from a Tempo2 predictor file.
        """
        args = {}
        coeffs = []
        i = 0
        while i < len(lines):
            parts = lines[i].split()
            if parts[0] == 'PSRNAME':
                args['psrname'] = parts[1]
            elif parts[0] == 'SITENAME':
                args['sitename'] = parts[1]
            elif parts[0] == 'TIME_RANGE':
                args['start_time'] = Time(
                    np.longdouble(parts[1]),
                    scale='utc',
                    format='pulsar_mjd_long',
                )
                args['end_time'] = Time(
                    np.longdouble(parts[2]),
                    scale='utc',
                    format='pulsar_mjd_long',
                )
            elif parts[0] == 'FREQ_RANGE':
                args['start_freq'] = np.longdouble(parts[1])*u.MHz
                args['end_freq'] = np.longdouble(parts[2])*u.MHz
            elif parts[0] == 'DISPERSION_CONSTANT':
                args['dispersion_constant'] = np.longdouble(parts[1])*u.MHz**2
            elif parts[0] == 'NCOEFF_TIME':
                ncoeff_time = int(parts[1])
            elif parts[0] == 'NCOEFF_FREQ':
                ncoeff_freq = int(parts[1])
            elif parts[0] == 'COEFFS':
                coeffs_freq = [np.longdouble(part) for part in parts[1:]]
                while len(coeffs_freq) < ncoeff_freq:
                    i += 1
                    parts = lines[i].split()
                    coeffs_freq.extend([np.longdouble(part) for part in parts])
                coeffs.append(coeffs_freq)
            i += 1
        coeffs = np.array(coeffs, dtype=np.longdouble)
        if coeffs.shape != (ncoeff_time, ncoeff_freq):
            raise ValueError(f'Number of coefficients ({self.coeffs.shape[0]}x{self.coeffs.shape[1]}) '
                             f'does not match specification ({self.ncoeff_time}x{self.ncoeff_freq}).')
        args['coeffs'] = coeffs
        return cls(**args)

    def __call__(self, time, freq, out_of_bounds='error'):
        """
        Evaluate the pulse phase at a given time and frequency.

        Parameters
        ----------
        time, freq: Time as an `astropy.Time` object, and frequency, as a `Quantity`.
                    Values should have shapes that can be broadcast together.
        out_of_bounds: How to treat out-of-bounds values. Possible values are:
                    'error' (raise an exception, the default),
                    'extrap' (attempt to extrapolate), and
                    'nan' (return not-a-number).
        """
        x, y, coeffs = self.chebval_args(time, freq, out_of_bounds)

        chebval = chebyshev.chebval2d(x, y, coeffs)
        phase = (self.dispersion_constant*freq**-2).to('').value + chebval
        return phase

    def f0(self, time, freq, out_of_bounds='error'):
        """
        Evaluate the pulse frequency at a given time and radio frequency.

        Parameters
        ----------
        time, freq: Time as an `astropy.Time` object, and frequency, as a `Quantity`.
                    Values should have shapes that can be broadcast together.
        out_of_bounds: How to treat out-of-bounds values. Possible values are:
                    'error' (raise an exception, the default),
                    'extrap' (attempt to extrapolate), and
                    'nan' (return not-a-number).
        """
        x, y, coeffs = self.chebval_args(time, freq, out_of_bounds)

        coeffs = chebyshev.chebder(coeffs, axis=0)
        chebval = chebyshev.chebval2d(x, y, coeffs)

        # convert to cycles/sec
        f0 = chebval * 2/(self.end_time.mjd_long - self.start_time.mjd_long) / 86400
        return f0

    def covers(self, time, freq):
        """
        Determine whether this model segment covers a given combination of time and frequency.

        Parameters
        ----------
        time, freq: Time as an `astropy.Time` object, and frequency, as a `Quantity`.
                    Values should have shapes that can be broadcast together.
        """
        time_covered = (time >= self.start_time) & (time <= self.end_time)
        min_freq = min(self.start_freq, self.end_freq)
        max_freq = max(self.start_freq, self.end_freq)
        freq_covered = (freq >= min_freq) & (freq <= max_freq)
        return time_covered & freq_covered

    def chebval_args(self, time, freq, out_of_bounds='error'):
        # Scale MJD and frequency to within [-1, 1]
        time_diff = time.mjd_long - self.start_time.mjd_long
        time_span = self.end_time.mjd_long - self.start_time.mjd_long
        freq_diff = freq - self.start_freq
        freq_span = self.end_freq - self.start_freq
        x = 2*time_diff/time_span - 1
        y = 2*(freq_diff/freq_span).to('').value - 1

        if np.any(~self.covers(time, freq)):
            if out_of_bounds == 'error':
                raise ValueError('Some points are out of bounds')
            elif out_of_bounds == 'extrap':
                pass
            elif out_of_bounds == 'nan':
                bad_times = (time_diff/time_span < 0) | (time_diff/time_span > 1)
                bad_freqs = (time_diff/time_span < 0) | (time_diff/time_span > 1)
                x[bad_times] = np.nan
                y[bad_freqs] = np.nan

        x, y = np.broadcast_arrays(x, y)

        # Account for differing conventions:
        # Tempo2 uses T_0(x) = 1/2, while Numpy uses T_0(x) = 1.
        coeffs = self.coeffs.copy()
        coeffs[0,:] /= 2
        coeffs[:,0] /= 2

        return x, y, coeffs

    def describe(self):
        """
        Return a string describing this model in the format of a Tempo2 predictor file.
        """
        description = "ChebyModel BEGIN\n"
        description += f"PSRNAME {self.psrname}\n"
        description += f"SITENAME {self.sitename}\n"
        description += (f"TIME_RANGE {np.format_float_scientific(self.start_time.mjd_long)} "
                        f"{np.format_float_scientific(self.end_time.mjd_long)}\n")
        description += (f"FREQ_RANGE {np.format_float_scientific(self.start_freq.to(u.MHz).value)} "
                        f"{np.format_float_scientific(self.end_freq.to(u.MHz).value)}\n")
        description += (f"DISPERSION_CONSTANT "
                        f"{np.format_float_scientific(self.dispersion_constant.to(u.MHz**2).value)}\n")
        description += f"NCOEFF_TIME {self.coeffs.shape[0]}\n"
        description += f"NCOEFF_FREQ {self.coeffs.shape[1]}\n"
        for coeff_set in self.coeffs:
            description += (f"COEFFS "
                            + " ".join(f"{np.format_float_scientific(coeff)}" for coeff in coeff_set)
                            + "\n")
        description += "ChebyModel END"
        return description

class ChebyModelSet:
    """
    A model of pulse phase as a function of frequency and time, defined by a set of segments
    each consisting of a 2-D Chebyshev Polynomial combined with a dispersion constant,
    as defined in a Tempo2 predictor file.
    """
    def __init__(self, segments):
        self.segments = segments

    @classmethod
    def parse(cls, lines):
        """
        Construct a ChebyModelSet object by reading lines from a Tempo2 predictor file.
        """
        segments = []
        for line in lines:
            parts = line.split()
            if len(parts) == 0:
                continue
            if parts[0] == 'ChebyModelSet':
                nsegments = int(parts[1])
            elif parts[0] == 'ChebyModel':
                if parts[1] == 'BEGIN':
                    model_lines = []
                elif parts[1] == 'END':
                    segments.append(ChebyModel.parse(model_lines))
            else:
                model_lines.append(line)
        if len(segments) != nsegments:
            raise ValueError(f'Number of segments ({len(segments)}) does not match specification ({nsegments}).')
        return cls(segments)

    def __call__(self, time, freq, out_of_bounds='error'):
        """
        Evaluate the pulse phase at a given time and frequency.

        Parameters
        ----------
        time, freq: Time as an `astropy.Time` object, and frequency, as a `Quantity`.
                    Values should have shapes that can be broadcast together.
        out_of_bounds: How to treat out-of-bounds values. Possible values are:
                    'error' (raise an exception, the default),
                    'extrap' (attempt to extrapolate), and
                    'nan' (return not-a-number).
        """
        time_broadcast, freq_broadcast = broadcast_time_freq(time, freq)
        phase = np.empty(np.broadcast_shapes(time.shape, freq.shape), dtype=np.float128)

        closest_segment = self.closest_segment(time, freq)
        for i, segment in enumerate(self.segments):
            sl = (closest_segment == i)
            phase[sl] = segment(time_broadcast[sl], freq_broadcast[sl], out_of_bounds)

        return phase[()] # turns 0d arrays into scalars, otherwise harmless

    def f0(self, time, freq, out_of_bounds='error'):
        """
        Evaluate the pulse frequency at a given time and radio frequency.

        Parameters
        ----------
        time, freq: Time as an `astropy.Time` object, and frequency, as a `Quantity`.
                    Values should have shapes that can be broadcast together.
        out_of_bounds: How to treat out-of-bounds values. Possible values are:
                    'error' (raise an exception, the default),
                    'extrap' (attempt to extrapolate), and
                    'nan' (return not-a-number).
        """
        time_broadcast, freq_broadcast = broadcast_time_freq(time, freq)
        phase = np.empty(np.broadcast_shapes(time.shape, freq.shape), dtype=np.float128)

        closest_segment = self.closest_segment(time, freq)
        for i, segment in enumerate(self.segments):
            sl = (closest_segment == i)
            f0[sl] = segment.f0(time_broadcast[sl], freq_broadcast[sl], out_of_bounds)

        return f0[()] # turns 0d arrays into scalars, otherwise harmless

    def covers(self, time, freq):
        """
        Determine whether this model covers a given combination of time and frequency.

        Parameters
        ----------
        time, freq: Time as an `astropy.Time` object, and frequency, as a `Quantity`.
                    Values should have shapes that can be broadcast together.
        """
        return np.any([segment.covers(time, freq) for segment in self.segments], axis=0)

    def closest_segment(self, time, freq):
        """
        Return an array containing the index of the segment covering each input.
        If multiple segments cover the input, the segment whose center is closest in time
        to the input will be returned.

        Parameters
        ----------
        time, freq: Time as an `astropy.Time` object, and frequency, as a `Quantity`.
                    Values should have shapes that can be broadcast together.
        """
        segment_centers = Time([
            segment.start_time + (segment.end_time - segment.start_time)/2 for segment in self.segments
        ])
        closest_segment = np.argmin(np.abs(time[..., np.newaxis].mjd_long - segment_centers.mjd_long), axis=-1)
        closest_segment, _ = np.broadcast_arrays(closest_segment, freq.value)
        return closest_segment

    def describe(self):
        """
        Return a string describing this model in the format of a Tempo2 predictor file.
        """
        description = f"ChebyModelSet {len(self.segments)} segments"
        for segment in self.segments:
            description += '\n'
            description += segment.describe()
        return description

def broadcast_time_freq(time, freq):
    jd1_broadcast, jd2_broadcast, freq_broadcast = np.broadcast_arrays(time.jd1, time.jd2, freq.value)
    time_broadcast = Time(jd1_broadcast, jd2_broadcast, format='jd')
    time_broadcast.format = time.format
    freq_broadcast = u.Quantity(freq_broadcast, freq.unit)
    return time_broadcast, freq_broadcast
