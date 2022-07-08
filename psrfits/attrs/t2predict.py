import numpy as np
from numpy.polynomial import chebyshev

class ChebyModel:
    def __init__(self, psrname, sitename, mjd_start, mjd_end, freq_start, freq_end,
                 dispersion_constant, coeffs):
        self.psrname = psrname
        self.sitename = sitename
        self.mjd_start = mjd_start
        self.mjd_end = mjd_end
        self.freq_start = freq_start
        self.freq_end = freq_end
        self.dispersion_constant = dispersion_constant
        self.coeffs = coeffs

    @classmethod
    def parse(cls, lines):
        args = {}
        coeffs = []
        for line in lines:
            parts = line.split()
            if parts[0] == 'PSRNAME':
                args['psrname'] = parts[1]
            elif parts[0] == 'SITENAME':
                args['sitename'] = parts[1]
            elif parts[0] == 'TIME_RANGE':
                args['mjd_start'] = np.longdouble(parts[1])
                args['mjd_end'] = np.longdouble(parts[2])
            elif parts[0] == 'FREQ_RANGE':
                args['freq_start'] = np.longdouble(parts[1])
                args['freq_end'] = np.longdouble(parts[2])
            elif parts[0] == 'DISPERSION_CONSTANT':
                args['dispersion_constant'] = np.longdouble(parts[1])
            elif parts[0] == 'NCOEFF_TIME':
                ncoeff_time = int(parts[1])
            elif parts[0] == 'NCOEFF_FREQ':
                ncoeff_freq = int(parts[1])
            elif parts[0] == 'COEFFS':
                coeffs.append([np.longdouble(part) for part in parts[1:]])
        coeffs = np.array(coeffs, dtype=np.longdouble)
        if coeffs.shape != (ncoeff_time, ncoeff_freq):
            raise ValueError(f'Number of coefficients ({self.coeffs.shape[0]}x{self.coeffs.shape[1]}) '
                             f'does not match specification ({self.ncoeff_time}x{self.ncoeff_freq}).')
        args['coeffs'] = coeffs
        return cls(**args)

    def __call__(self, mjd, freq, check_bounds=True):
        x, y, coeffs = self.chebval_args(mjd, freq, check_bounds)

        chebval = chebyshev.chebval2d(x, y, coeffs)
        phase = self.dispersion_constant*freq**-2 + chebval
        return phase

    def f0(self, mjd, freq, check_bounds=True):
        x, y, coeffs = self.chebval_args(mjd, freq, check_bounds)

        coeffs = chebyshev.chebder(coeffs, axis=0)
        chebval = chebyshev.chebval2d(x, y, coeffs)
        f0 = chebval * 2/(self.mjd_end - self.mjd_start) / 86400 # convert to cycles/sec
        return f0

    def covers(self, mjd, freq):
        return ((self.mjd_start <= mjd <= self.mjd_end) and
                (min(self.freq_start, self.freq_end) <= freq <= max(self.freq_start, self.freq_end)))

    def chebval_args(self, mjd, freq, check_bounds=True):
        if check_bounds and (mjd < self.mjd_start or mjd > self.mjd_end):
            raise ValueError('MJD out of bounds.')
        if check_bounds and (freq < min(self.freq_start, self.freq_end) or freq > max(self.freq_start, self.freq_end)):
            raise ValueError('Frequency out of bounds.')

        # Scale MJD and frequency to within [-1, 1]
        x = 2*(mjd - self.mjd_start)/(self.mjd_end - self.mjd_start) - 1
        y = 2*(freq - self.freq_start)/(self.freq_end - self.freq_start) - 1

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
        description += f"TIME_RANGE {np.format_float_scientific(self.mjd_start)} {np.format_float_scientific(self.mjd_end)}\n"
        description += f"FREQ_RANGE {np.format_float_scientific(self.freq_start)} {np.format_float_scientific(self.freq_end)}\n"
        description += f"DISPERSION_CONSTANT {np.format_float_scientific(self.dispersion_constant)}\n"
        description += f"NCOEFF_TIME {self.coeffs.shape[0]}\n"
        description += f"NCOEFF_FREQ {self.coeffs.shape[1]}\n"
        for coeff_set in self.coeffs:
            description += f"COEFFS " + " ".join(f"{np.format_float_scientific(coeff)}" for coeff in coeff_set) + "\n"
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

    def __call__(self, mjd, freq, extrap=False):
        segment = self.covering_segment(mjd, freq, extrap)
        return segment(mjd, freq, check_bounds=(not extrap))

    def f0(self, mjd, freq, extrap=False):
        segment = self.covering_segment(mjd, freq, extrap)
        return segment.f0(mjd, freq, check_bounds=(not extrap))

    def covers(self, mjd, freq):
        return any(segment.covers(mjd, freq) for segment in self.segments)

    def covering_segment(self, mjd, freq, extrap=False):
        covering_segments = [segment.covers(mjd, freq) for segment in self.segments]
        if not any(covering_segments):
            if extrap:
                mjd_diffs = [mjd - np.abs((segment.mjd_start + segment.mjd_end)/2) for segment in segments]
                segment = relevant_segments[np.argmin(mjd_diffs)]
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
