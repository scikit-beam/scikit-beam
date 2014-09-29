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
from numpy import sin, cos
import scipy.signal
from collections import deque
from nsls2.constants import calibration_standards
from nsls2.feature import (filter_peak_height, peak_refinement,
                           refine_log_quadratic)
from nsls2.core import (pixel_to_phi, pixel_to_radius,
                        pairwise, bin_edges_to_centers, bin_1D,
                        bin_image_to_1D)
from nsls2.image import find_ring_center_acorr_1D


def estimate_d_blind(name, wavelength, bin_centers, ring_average,
               window_size, max_peak_count, thresh):
    """ Estimate the sample-detector distance

    Given a radially integrated calibration image return an estimate for
    the sample-detector distance.  This function does not require a
    rough estimate of what d should be.

    For the peaks found the detector-sample distance is estimated via

    .. math ::

        D = \\frac{r}{\\tan 2\\theta}

    where :math:`r` is the distance in mm from the calibrated center
    to the ring on the detector and :math:`D` is the distance from
    the sample to the detector.

    Parameters
    ----------
    name : str
        The name of the calibration standard.  Used to look up the
        expected peak location
        For valid options, see the name attribute on this function

    wavelength : float
        The wavelength of scattered x-ray in nm

    bin_centers : array
        The distance from the calibrated center to the center of
        the ring's annulus in mm

    ring_average : array
        The average intensity in the given ring of a azimuthally integrated
        powder pattern.  In counts [arb]

    window_size : int
        The number of elements on either side of a local maximum to
        use for locating and refining peaks.  Candidates are identified
        as a relative maximum in a window sized (2*window_size + 1) and
        the same window is used for fitting the peaks to refine the location.

    max_peak_count : int
        Use at most this many peaks

    thresh : float
        Fraction of maximum peak height
    Returns
    -------
    dist_sample : float
        The detector-sample distance in mm.  This is the mean of the estimate
        from all of the peaks used.

    std_dist_sample : float
        The standard deviation of d computed from the peaks used.

    """

    # get the calibration standard
    cal = calibration_standards[name]
    # find the local maximums
    cands = scipy.signal.argrelmax(ring_average, order=window_size)[0]
    # filter local maximums by size
    cands = filter_peak_height(ring_average, cands,
                               thresh*np.max(ring_average), window=window_size)
    # TODO insert peak identification validation.  This might be better than
    # improving the threshold value.
    # refine the locations of the peaks
    peaks_x, peaks_y = peak_refinement(bin_centers, ring_average, cands,
                                       window_size, refine_log_quadratic)
    # compute tan(2theta) for the expected peaks
    tan2theta = np.tan(cal.convert_2theta(wavelength))
    # figure out how many peaks we can look at
    slc = slice(0, np.min([len(tan2theta), len(peaks_x), max_peak_count]))
    # estimate the sample-detector distance for each of the peaks
    d_array = (peaks_x[slc] / tan2theta[slc])
    return np.mean(d_array), np.std(d_array)

# Set an attribute for the calibration names that are valid options.  This
# attribute also aids in autowrapping into VisTrails
estimate_d_blind.name = list(calibration_standards)


def refine_center(image, calibrated_center, pixel_size, phi_steps, max_peaks,
                  thresh, window_size,
                  nx=None, min_x=None, max_x=None):
    """Refines the location of the center of the beam.

    This relies on being able to see the whole powder pattern.

    Parameters
    ----------
    image : ndarray
        The image
    calibrated_center : tuple
        (row, column) the estimated center
    pixel_size : tuple
        (pixel_height, pixel_width)
    phi_steps : int
        How many regions to split the ring into, should be >10
    max_peaks : int
        Number of rings to look it
    thresh : float
        Fraction of maximum peak height
    window_size : int, optional
        The window size to use (in bins) to use when refining peaks
    nx : int, optional
        Number of bins to use for radial binning
    min_x : float, optional
        The minimum radius to use for radial binning
    max_x : float, optional
        The maximum radius to use for radial binning

    Returns
    -------
    calibrated_center : tuple
        The refined calibrated center.
    """
    if nx is None:
        nx = int(np.mean(image.shape) * 2)

    phi = pixel_to_phi(image.shape, calibrated_center, pixel_size).ravel()
    r = pixel_to_radius(image.shape, calibrated_center, pixel_size).ravel()
    I = image.ravel()

    phi_steps = np.linspace(-np.pi, np.pi, phi_steps, endpoint=True)
    out = deque()
    for phi_start, phi_end in pairwise(phi_steps):
        mask = (phi <= phi_end) * (phi > phi_start)
        out.append(bin_1D(r[mask], I[mask],
                          nx=nx, min_x=min_x, max_x=max_x))
    out = list(out)

    ring_trace = []
    for bins, b_sum, b_count in out:
        mask = b_sum > 10
        avg = b_sum[mask] / b_count[mask]
        bin_centers = bin_edges_to_centers(bins)[mask]

        cands = scipy.signal.argrelmax(avg, order=window_size)[0]
        # filter local maximums by size
        cands = filter_peak_height(avg, cands, thresh*np.max(avg),
                                           window=window_size)
        ring_trace.append(bin_centers[cands[:max_peaks]])

    tr_len = [len(rt) for rt in ring_trace]
    mm = np.min(tr_len)
    ring_trace = np.vstack([rt[:mm] for rt in ring_trace]).T

    mean_dr = np.mean(ring_trace - np.mean(ring_trace, axis=1, keepdims=True),
                      axis=0)

    phi_centers = bin_edges_to_centers(phi_steps)

    delta = np.mean(np.diff(phi_centers))
    # this is doing just one term of a Fourier series
    # note that we have to convert _back_ to pixels from real units
    # TODO do this with better integration/handle repeat better
    col_shift = (np.sum(np.sin(phi_centers) * mean_dr) *
                 delta / (np.pi * pixel_size[1]))
    row_shift = (np.sum(np.cos(phi_centers) * mean_dr) *
                 delta / (np.pi * pixel_size[0]))

    return tuple(np.array(calibrated_center) +
                 np.array([row_shift, col_shift]))


def powder_auto_calibrate(img, name, wavelength, pixel_size):
    """
    Automatically find the beam center, detector tilt, and sample distance

    .. warning:: This function is not finished and the API may change

    .. note:: Currently the *full* rings need to be visible.

    Parameters
    ----------
    img : ndarray
        The calibration image.

    name : str
        The name of the calibration sample.  The known standards are
        stored in `nsls2.constants.calibration_standards`.

    wavelength : float
        x-ray wave length in angstroms

    pixel_size : tuple
        The (height, width) pitch of the detector pixels in mm

    Returns
    -------
    D : float
       The sample-to-detector distance in mm
       (more accuratly, same units as `pixel_size`)

    D_error : float
       Estimate of error in D

    center : tuple
        (row, col) calibrated beam center in pixels

    center_error : tuple
        Estimated error in the center location.

    tilt : tuple
        (phi1, phi2) in radians giving the direction of the tilt and
        the degree of tilt.  See `tilt_detector` and `untilt_detector`

    tilt_error : tupel
        Estimated error in the tilt angles




    """

    res = find_ring_center_acorr_1D(img)
    center = refine_center(img, res, pixel_size, 25, 5,
                         thresh=0.1, window_size=5)
    bins, sums, counts = bin_image_to_1D(img,
                                         center,
                                         pixel_to_radius,
                                         pixel_to_1D_kwarg={'pixel_size':
                                                            pixel_size},
                                         bin_num=5000)

    mask = counts > 10
    bin_centers = bin_edges_to_centers(bins)[mask]
    ring_averages = sums[mask] / counts[mask]

    d_mean, d_std = estimate_d_blind(name, wavelength, bin_centers,
                                 ring_averages, 5, 7, thresh=0.03)

    tilt = None
    center_error = None
    tilt = None
    tilt_error = None
    return d_mean, d_std, center, center_error, tilt, tilt_error

powder_auto_calibrate.name = list(calibration_standards)


def tilt_coords(phi1, phi2, row, col):
    """
    Returns the measured coordinates on the detector if it is tilted
    by an angle phi2 against the axis which is perpendicular to
    the line given by phi1 for a set of true coordinates.

    The tilt axis goes through the origin.  The input data must be shifted
    so that the center is at (0, 0) prior to calling this function.

    Parameters
    ----------
    phi1 : float
        Rotation of the tilt axis.  The tilt angle is about
        the line given by phi1.

        Put another way, this vector points along the minor axis
        if the ellipse.

        In radians

    phi2 : float
        The angle of the tilt around the line defined by phi1.

        In radians

    row : array
       True row positions to tilt

    col : array
       True column positions to tilt

    Returns
    -------
    row : array
       The deformed row positions

    col : array
       The deformed column positions


    See Also
    --------
    untilt_coords : inverse function
    """
    row = np.asarray(row)
    col = np.asarray(col)

    c1 = cos(phi1)
    c2 = cos(phi2)
    s1 = sin(phi1)

    new_row = (c1*c1/c2 + s1*s1) * row + (-c1*s1/c2 + s1*c1) * col
    new_col = (-c1*s1/c2 + c1*s1) * row + (s1*s1/c2 + c1*c1) * col

    return new_row, new_col


def untilt_coords(phi1, phi2, row, col):
    """
    Returns the True coordinates on the detector if it is tilted
    by an angle phi2 against the axis which is perpendicular to
    the line given by phi1 for a set of measured coordinates.

    The tilt axis goes through the origin.  The input data must be shifted
    so that the center is at (0, 0) prior to calling this function.

    Parameters
    ----------
    phi1 : float
        Rotation of the tilt axis.  The tilt angle is about
        the axis perpendicular to the line given by phi1.

        Put another way, this vector points along the minor axis
        if the ellipse.

        In radians

    phi2 : float
        The angle of the tilt about the line perpendicular to the
        vector defined by phi1.

        In radians

    row : array
       The row positions to tilt

    col : array
       The column positions to tilt

    Returns
    -------
    row : array
       The deformed row positions

    col : array
       The deformed column positions

    See Also
    --------
    tilt_coords : inverse function

    """
    c1 = cos(phi1)
    c2 = cos(phi2)
    s1 = sin(phi1)

    new_row = (c1*c1*c2 + s1*s1) * row + (-c1*s1*c2 + s1*c1) * col
    new_col = (-c1*s1*c2 + c1*s1) * row + (s1*s1*c2 + c1*c1) * col

    return new_row, new_col


def tilt_angles_to_coefs(r, phi1, phi2):
    """
    Compute the coefficients for the Fourier expansion of a
    circle on a tilted detector.

    This is mostly useful for testing

    Parameters
    ----------
    r : float
        The radius of the un-distorted circle

    phi1 : float
        The first tilt angle in radians.  See `tilt_coords`

    phi2 : float
        The second tilt angle

    Returns
    -------
    r0 : float
        The constant coefficient in the Fourier expansion of r*r

    a1 : float
        The coefficent on the `cos(2 * chi)` in the Fourier expansion

    a2 : float
        The coefficent on the `sin(2 * chi)` in the Fourier expansion
    """
    c2 = cos(phi2)
    pre_factor = (1 - 1 / (c2 * c2))
    r0 = r*r * .5 * ((1 / (c2*c2) + 1))
    a1 = - r*r * .5 * cos(phi1 * 2) * pre_factor
    a2 = r*r * .5 * sin(phi1 * 2) * pre_factor
    return r0, a1, a2


def data_to_coefs(chi, r_sq):
    """
    Given `r^2(\chi)` compute the Fourier coefficients for a ring.

    Parameters
    ----------
    chi : array
        The angles at which the ring radius is sampled

    r_sq : array
        The radius squared of the ring

    Returns
    -------
    r0 : float
        The constant term in the Fourier series

    a1 : float
        The coefficient on cos(2*chi) in the Fourier series

    a2 : float
        The coefficient on sin(2*chi) in the Fourier series
    """
    # make sure everything really is an array
    chi = np.asarray(chi)
    r_sq = np.asarray(r_sq)
    # compute the mean (constant term)
    r0 = np.mean(r_sq)

    # compute the 2chi coefficients
    # TODO replace this with more accurate integration
    delta = np.mean(np.diff(chi))
    a1 = np.sum(cos(2 * chi) * r_sq) * delta / np.pi
    a2 = np.sum(sin(2 * chi) * r_sq) * delta / np.pi

    return r0, a1, a2


def coefs_to_phi1(a1, a2):
    """
    Given the coefficients on the 2chi terms of the
    Fourier series compute the first tilt angle.

    Parameters
    ----------
    a1 : float
        The coefficient on cos(2*chi) in the Fourier series

    a2 : float
        The coefficient on sin(2*chi) in the Fourier series

    Returns
    -------
    phi1 : float
        The first tilt angle (the direction of the minor axis)
    """
    return - 0.5 * np.arctan2(a2, a1)


def coefs_to_phi2_1(a1, r0, phi1):
    """
    Given the constant and cos(2chi) coefficients and
    the first tilt angle, compute the second tilt angle

    Parameters
    ----------
    a1 : float
        The coefficient on cos(2*chi) in the Fourier series

    r0 : float
        The constant term in the Fourier series

    phi1 : float
        The first tilt angle

    Returns
    -------
    phi2 : float
        The second tilt angle
    """
    P = -a1 / (r0 * cos(2 * phi1))
    return np.arccos(np.sqrt((1 + P) / (1 - P)))


def compute_phi2_2(a2, r0, phi1):
    """
    Given the constant and cos(2chi) coefficients and
    the first tilt angle, compute the second tilt angle

    Parameters
    ----------
    a2 : float
        The coefficient on sin(2*chi) in the Fourier series

    r0 : float
        The constant term in the Fourier series

    phi1 : float
        The first tilt angle

    Returns
    -------
    phi2 : float
        The second tilt angle

    .. warning this return nan if phi1 == 0 due to a `1/sin(phi1)` term
    """
    P = a2 / (r0 * sin(2 * phi1))
    return np.arccos(np.sqrt((1 + P) / (1 - P)))


def coefs_to_r(r0, phi2):
    """

    Given the constant term and phi2 value, compute the radius of the
    un-tilted circle (which is also the minor-axis)

    Parameters
    ----------
    r0 : float
        The constant term in the Fourier series

    phi2 : float
        The second tilt angle

    Returns
    -------
    r : float
        The radius of the un-tilted circle/the minor axis

    """
    c2 = cos(phi2)
    return np.sqrt(r0 / (.5 * ((1 / (c2*c2) + 1))))


def coefs_to_params(r0, a1, a2):
    """Fourier series coefficients to tilt angles

    Given the coefficients from the Fourier series compute the
    tilt angles.

    Parameters
    ----------
    r0 : float
        The constant coefficient in the Fourier expansion of r*r

    a1 : float
        The coefficent on the `cos(2 * chi)` in the Fourier expansion

    a2 : float
        The coefficent on the `sin(2 * chi)` in the Fourier expansion

    Returns
    -------
    r : float
        The radius of the un-distorted circle

    phi1 : float
        The first tilt angle in radians.  See `tilt_coords`

    phi2 : float
        The second tilt angle

    See Also
    --------
    `tilt_angles_to_coefs`
    """
    phi1 = coefs_to_phi1(a1, a2)
    phi2 = coefs_to_phi2_1(a1, r0, phi1)
    r = coefs_to_r(r0, phi2)

    return r, phi1, phi2
