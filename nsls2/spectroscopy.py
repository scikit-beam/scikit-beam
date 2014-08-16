# Copyright (c) Brookhaven National Lab 2O14
# All rights reserved
# BSD License
# See LICENSE for full text
"""
This module is for spectroscopy specific tools (spectrum fitting etc).
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from six.moves import zip
import numpy as np
from scipy.integrate import simps
import logging


def fit_quad_to_peak(x, y):
    """
    Fits a quadratic to the data points handed in
    to the from y = b[0](x-b[1])**2 + b[2] and R2
    (measure of goodness of fit)

    Parameters
    ----------
    x : ndarray
        locations
    y : ndarray
        values

    Returns
    -------
    b : tuple
       coefficients of form y = b[0](x-b[1])**2 + b[2]

    R2 : float
      R2 value

    """

    lenx = len(x)

    # some sanity checks
    if lenx < 3:
        raise Exception('insufficient points handed in ')
    # set up fitting array
    X = np.vstack((x ** 2, x, np.ones(lenx))).T
    # use linear least squares fitting
    beta, _, _, _ = np.linalg.lstsq(X, y)

    SSerr = np.sum(np.power(np.polyval(beta, x) - y, 2))
    SStot = np.sum(np.power(y - np.mean(y), 2))
    # re-map the returned value to match the form we want
    ret_beta = (beta[0],
                -beta[1] / (2 * beta[0]),
                beta[2] - beta[0] * (beta[1] / (2 * beta[0])) ** 2)

    return ret_beta, 1 - SSerr / SStot


def align_and_scale(energy_list, counts_list, pk_find_fun=None):
    """

    Parameters
    ----------
    energy_list : iterable of ndarrays
        list of ndarrays with the energy of each element

    counts_list : iterable of ndarrays
        list of ndarrays of counts/element

    pk_find_fun : function or None
       A function which takes two ndarrays and returns parameters
       about the largest peak.  If None, defaults to `find_largest_peak`.
       For this demo, the output is (center, height, width), but this sould
       be pinned down better.

    Returns
    -------
    out_e : list of ndarray
       The aligned/scaled energy arrays

    out_c : list of ndarray
       The count arrays (should be the same as the input)
    """
    if pk_find_fun is None:
        pk_find_fun = find_larest_peak

    base_sigma = None
    out_e, out_c = [], []
    for e, c in zip(energy_list, counts_list):
        E0, max_val, sigma = pk_find_fun(e, c)
        print(E0, max_val, sigma)
        if base_sigma is None:
            base_sigma = sigma
        out_e.append((e - E0) * base_sigma / sigma)
        out_c.append(c)

    return out_e, out_c


def find_larest_peak(X, Y, window=5):
    """
    Finds and estimates the location, width, and height of
    the largest peak. Assumes the top of the peak can be
    approximated as a Gaussian.  Finds the peak properties
    using least-squares fitting of a parabola to the log of
    the counts.

    The region around the peak can be approximated by
    Y = Y0 * exp(- (X - X0)**2 / (2 * sigma **2))

    Parameters
    ----------
    X : ndarray
       The independent variable

    Y : ndarary
      Dependent variable sampled at positions X

    window : int, optional
       The size of the window around the maximum to use
       for the fitting


    Returns
    -------
    X0 : float
        The location of the peak

    Y0 : float
        The magnitude of the peak

    sigma : float
        Width of the peak
    """

    # make sure they are _really_ arrays
    X = np.asarray(X)
    Y = np.asarray(Y)

    # get the bin with the largest number of counts
    j = np.argmax(Y)
    roi = slice(np.max(j - window, 0),
                j + window + 1)

    (w, X0, Y0), R2 = fit_quad_to_peak(X[roi],
                                        np.log(Y[roi]))

    return X0, np.exp(Y0), 1/np.sqrt(-2*w)

def integrate_ROI_spectrum(bin_edges, counts, x_min, x_max):
    """Integrate region(s) of histogram.

    If `x_min` and `x_max` are arrays/lists they must be equal in
    length. The values contained in the 'x_value_array' must be
    monotonic (up or down).  The returned value is the sum
    of all the regions and a single scalar value is returned.

    `bin_edges` is an array of the left edges and the final right
    edges of the bins.  `counts` is the value in each of those bins.

    The bins who's centers fall with in the integration limits are
    included in the sum.


    Parameters
    ----------
    bin_edges : array
        Independent variable, any unit.

        Must be one longer in length than counts

    counts : array
        Dependent variable, any units

    x_min : float or array
        The lower edge of the integration region(s).

    x_max : float or array
        The upper edge of the integration region(s).

    Returns
    -------
    float
        The totals integrated value in same units as `counts`

    """
    bin_edges = np.asarray(bin_edges)
    return integrate_ROI(bin_edges[:-1] + np.diff(bin_edges),
                         counts, x_min, x_max)


def integrate_ROI(x_value_array, counts, x_min, x_max):
    """Integrate region(s) of .

    If `x_min` and `x_max` are arrays/lists they must be equal in
    length. The values contained in the 'x_value_array' must be
    monotonic (up or down).  The returned value is the sum
    of all the regions and a single scalar value is returned.

    This function assumes that `counts` is a function of
    `x_value_array` sampled at `x_value_array`.

    Parameters
    ----------
    x_value_array : array
        Independent variable, any unit

    counts : array
        Dependent variable, any units

    x_min : float or array
        The lower edge of the integration region(s).

    x_max : float or array
        The upper edge of the integration region(s).

    Returns
    -------
    float
        The totals integrated value in same units as `counts`
    """
    # make sure x_value_array (x-values) and counts (y-values) are arrays
    x_value_array = np.asarray(x_value_array)
    counts = np.asarray(counts)

    if x_value_array.shape != counts.shape:
        raise ValueError("Inputs must be same size")

    #use np.sign() to obtain array which has evaluated sign changes in all diff
    #in input x_value array. Checks and tests are then run on the evaluated
    #sign change array.
    eval_x_arr_sign = np.sign(np.diff(x_value_array))

    #check to make sure no outliers exist which violate the monotonically
    #increasing requirement, and if exceptions exist, then error points to the
    #location within the source array where the exception occurs.
    if not np.all(eval_x_arr_sign == eval_x_arr_sign[0]):
        error_locations = np.where(eval_x_arr_sign <= 0)
        raise ValueError("Independent variable must be monotonically "
                         "increasing. Erroneous values found at x-value "
                         "array index locations: {0}".format(error_locations))

    # check whether the sign of all diff measures are negative in the
    # x_value_array. If so, then the input array for both x_values and
    # count are reversed so that they are positive, and monotonically increase
    # in value
    if eval_x_arr_sign[0] == -1:
        x_value_array = x_value_array[::-1]
        counts = counts[::-1]
        logging.debug("Input values for 'x_value_array' were found to be monotonically "
                "decreasing. The 'x_value_array' and 'counts' arrays have been"
                " reversed prior to integration.")

    # up-cast to 1d and make sure it is flat
    x_min = np.atleast_1d(x_min).ravel()
    x_max = np.atleast_1d(x_max).ravel()

    # verify that the number of minimum and maximum boundary values are equal
    if len(x_min) != len(x_max):
        raise ValueError("integration bounds must have same lengths")

    # verify that the specified minimum values are actually less than the sister
    # maximum value, and raise error if any minimum value is actually greater
    #than the sister maximum value.
    if np.any(x_min >= x_max):
        raise ValueError("All lower integration bounds must be less than "
                         "upper integration bounds.")

    # check to make sure that all specified minimum and maximum values are
    # actually contained within the extents of the independent variable array
    if np.any(x_min < x_value_array[0]):
        error_locations = np.where(x_min < x_value_array[0])
        raise ValueError("Specified lower integration boundary values "
                         "are outside the spectrum range. All minimum "
                         "integration boundaries must be greater than, or "
                         "equal to the lowest value in spectrum range. The "
                         "erroneous x_min array indices are: {0}".format(error_locations))
    if np.any(x_max > x_value_array[-1]):
        error_locations =  np.where(x_max > x_value_array[-1])
        raise ValueError("Specified upper integration boundary values "
                         "are outside the spectrum range. All maximum "
                         "integration boundary values must be less "
                         "than, or equal to the highest value in the spectrum "
                         "range. The erroneous x_max array indices are: "
                         "{0}".format(error_locations))

    # find the bottom index of each integration bound
    bottom_indx = x_value_array.searchsorted(x_min)
    # find the top index of each integration bound
    # NOTE: +1 required for correct slicing for integration function
    top_indx = x_value_array.searchsorted(x_max) + 1

    # set up temporary variables
    accum = 0
    # integrate each region
    for bot, top in zip(bottom_indx, top_indx):
        # Note: If an odd number of intervals is specified, then the
        # even='avg' setting calculates and averages first AND last
        # N-2 intervals using trapezoidal rule.
        # If calculation speed become an issue, then consider changing
        # setting to 'first', or 'last' in which case trap rule is only
        # applied to either first or last N-2 intervals.
        accum += simps(counts[bot:top], x_value_array[bot:top], even='avg')

    return accum
