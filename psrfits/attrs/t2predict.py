import numpy as np
from numpy.polynomial import chebyshev
from astropy.time import Time
import astropy.units as u
from pint import PulsarMJD

class ChebyModel:
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
                args['start_time'] = Time(np.longdouble(parts[1]), format='pulsar_mjd_long')
                args['end_time'] = Time(np.longdouble(parts[2]), format='pulsar_mjd_long')
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
        x, y, coeffs = self.chebval_args(time, freq, out_of_bounds)

        chebval = chebyshev.chebval2d(x, y, coeffs)
        phase = (self.dispersion_constant*freq**-2).to('').value + chebval
        return phase

    def f0(self, time, freq, out_of_bounds='error'):
        x, y, coeffs = self.chebval_args(time, freq, out_of_bounds)

        coeffs = chebyshev.chebder(coeffs, axis=0)
        chebval = chebyshev.chebval2d(x, y, coeffs)

        # convert to cycles/sec
        f0 = chebval * 2/(self.end_time.mjd_long - self.start_time.mjd_long) / 86400
        return f0

    def covers(self, time, freq):
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
        description = "ChebyModel BEGIN\n"
        description += f"PSRNAME {self.psrname}\n"
        description += f"SITENAME {self.sitename}\n"
        description += (f"TIME_RANGE {np.format_float_scientific(self.start_time.mjd_long)} "
                        f"{np.format_float_scientific(self.end_time.mjd_long)}\n")
        description += (f"FREQ_RANGE {np.format_float_scientific(self.start_freq.to(u.MHz).value)} "
                        f"{np.format_float_scientific(self.end_freq.to(u.MHz).value)}\n")
        description += f"DISPERSION_CONSTANT {np.format_float_scientific(self.dispersion_constant)}\n"
        description += f"NCOEFF_TIME {self.coeffs.shape[0]}\n"
        description += f"NCOEFF_FREQ {self.coeffs.shape[1]}\n"
        for coeff_set in self.coeffs:
            description += (f"COEFFS "
                            + " ".join(f"{np.format_float_scientific(coeff)}" for coeff in coeff_set)
                            + "\n")
        description += "ChebyModel END"
        return description

class ChebyModelSet:
    def __init__(self, segments):
        self.segments = segments

    @classmethod
    def parse(cls, lines):
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
        segment = self.covering_segment(time, freq, out_of_bounds)
        return segment(time, freq, out_of_bounds)

    def f0(self, time, freq, out_of_bounds='error'):
        segment = self.covering_segment(time, freq, out_of_bounds)
        return segment.f0(time, freq, out_of_bounds)

    def covers(self, time, freq):
        return np.any([segment.covers(time, freq) for segment in self.segments], axis=0)

    def covering_segment(self, time, freq, extrap=False):
        segment_centers = Time([
            segment.start_time + (segment.end_time - segment.start_time)/2 for segment in self.segments
        ])
        closest_segment = np.argmin(np.abs(time[..., np.newaxis].mjd_long - segment_centers.mjd_long), axis=-1)
        covered = np.array([segment.covers(mjd, freq) for segment in self.segments])
        covering_segments = np.where(covered)
        if not any(covering_segments):
            if extrap:
                mjd_diffs = [mjd - np.abs((segment.mjd_start + segment.mjd_end)/2) for segment in self.segments]
                segment = self.segments[np.argmin(mjd_diffs)]
            else:
                raise ValueError('MJD and frequency not covered by any segments.')
        elif sum(covering_segments) > 1:
            # Multiple segments cover this MJD and frequency
            relevant_segments = [segment for segment, covers in zip(self.segments, covering_segments) if covers]
            mjd_diffs = [mjd - np.abs((segment.mjd_start + segment.mjd_end)/2) for segment in relevant_segments]
            segment = relevant_segments[np.argmin(mjd_diffs)]
        else:
            segment = self.segments[covering_segments.index(True)]
        return segment

    def describe(self):
        description = f"ChebyModelSet {len(self.segments)} segments"
        for segment in self.segments:
            description += '\n'
            description += segment.describe()
        return description
