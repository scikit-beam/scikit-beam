# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Redistribution and use in source and binary forms, with or without   #
# modification, are permitted provided that the following conditions   #
# are met:                                                             #
#                                                                      #
# * Redistributions of source code must retain the above copyright     #
#   notice, this list of conditions and the following disclaimer.      #
#                                                                      #
# * Redistributions in binary form must reproduce the above copyright  #
#   notice this list of conditions and the following disclaimer in     #
#   the documentation and/or other materials provided with the         #
#   distribution.                                                      #
#                                                                      #
# * Neither the name of the Brookhaven Science Associates, Brookhaven  #
#   National Laboratory nor the names of its contributors may be used  #
#   to endorse or promote products derived from this software without  #
#   specific prior written permission.                                 #
#                                                                      #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS  #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT    #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS    #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE       #
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,           #
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR   #
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)   #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,  #
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OTHERWISE) ARISING   #
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE   #
# POSSIBILITY OF SUCH DAMAGE.                                          #
########################################################################
"""
This is the module for calibration functions and data
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
import scipy.signal
from nsls2.constants import calibration_standards
from nsls2.feature import (filter_peak_height, peak_refinement,
                           refine_log_quadratic)


def estimate_d_blind(name, wavelength, bin_centers, ring_average,
               window_size, threshold, max_peak_count=None):
    """ Estimate the sample-detector distance

    Given a radially integrated calibration image return an estimate for
    the sample-detector distance.  This function does not require a
    rough estimate of what d should be.

    For the peaks found the detector-sample distance is estimated via
    .. math ::

        d = \\frac{m}{\\tan 2\\theta}

    where :math:`m` is the distance in mm from the calibrated center
    to the ring on the detector.

    Parameters
    ----------
    name : str
        The name of the calibration standard.  Used to look up the
        expected peak location

    wavelength : float
        The wavelength of scattered x-ray in nm

    bin_centers : array
        The distance from the calibrated center to the center of
        the ring's annulus in mm

    ring_average : array
        The average intensity in the given ring.  In counts [arb]

    window_size : int
        The number of elements on either side of a local maximum to
        use for locating and refining peaks.  Candidates are identified
        as a relative maximum in a window sized (2*window_size + 1) and
        the same window is used for fitting the peaks to refine the location.

    threshold : float
        The minimum range a peak needs to have in the window to be accepted
        as a real peak.  This is used to filter out the (many) spurious local
        maximum which can be found.

    max_peak_count : int, optional
        Use at most this many peaks

    Returns
    -------
    dist_sample : float
        The detector-sample distance in mm.  This is the mean of the estimate
        from all of the peaks used.

    std_dist_sample : float
        The standard deviation of the d estimated by each of the peaks

    """
    if max_peak_count is None:
        max_peak_count = np.iinfo(int).max
    # TODO come up with way to estimate threshold blind, maybe otsu

    # get the calibration standard
    cal = calibration_standards[name]
    # find the local maximums
    cands = scipy.signal.argrelmax(ring_average, order=window_size)[0]
    # filter local maximums by size
    cands = filter_peak_height(ring_average, cands,
                               threshold, window=window_size)
    # TODO insert peak identification validation.  This might be better than
    # improving the threshold value.

    # refine the locations of the peaks
    peaks_x, peaks_y = peak_refinement(bin_centers, ring_average, cands,
                                       window_size, refine_log_quadratic)
    # compute tan(2theta) for the expected peaks
    tan2theta = np.tan(cal.convert_2theta(wavelength))
    # figure out how many peaks we can look at
    slc = slice(0, np.min([len(tan2theta), len(peaks_x), max_peak_count]))
    # estimate the sample-
    d_array = (peaks_x[slc] / tan2theta[slc])

    return np.mean(d_array), np.std(d_array)
