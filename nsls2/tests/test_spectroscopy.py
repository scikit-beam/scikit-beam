from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from matplotlib import pyplot as plt
import numpy as np
from nose.tools import assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal

from nsls2.spectroscopy import align_and_scale, integrate_ROI


def synthetic_data(E, E0, sigma, alpha, k, beta):
    """
    return synthetic data of the form
    d = alpha * e ** (-(E - e0)**2 / (2 * sigma ** 2) + beta * sin(k * E)

    Parameters
    ----------
    E : ndarray
        The energies to compute values at

    E0 : float
       Location of the peak

    sigma : float
       Width of the peak

    alpha : float
       Height of peak

    k : float
       Frequency of oscillations

    beta : float
       Magnitude of oscillations
    """
    return (alpha * np.exp(-(E - E0)**2 / (2 * sigma**2)) +
            beta * (1 + np.sin(k * E)))


def test_align_and_scale_smoketest():
    # does nothing but call the function

    # make data
    E = np.linspace(0, 50, 1000)
    # this is not efficient for large lists, but quick and dirty
    e_list = []
    c_list = []
    for j in range(25, 35, 2):
        e_list.append(E)
        c_list.append(synthetic_data(E,
                                     j + j / 100,
                                     j / 10, 1000,
                                     2*np.pi * 6/50, 60))
    # call the function
    e_cor_list, c_cor_list = align_and_scale(e_list, c_list)


def test_integrate_ROI_errors():
    E = np.arange(100)
    C = np.ones_like(E)

    # limits out of order
    # NOTE: this will now get fixed in the code and will not raise exception.
    #assert_raises(ValueError, integrate_ROI, E, C, 32, 2)
    
    #Min boundary greater than max boundary.
    assert_raises(ValueError, integrate_ROI, E, C,
                  [32, 1], [2, 10])
    # bottom out of range
    assert_raises(ValueError, integrate_ROI, E, C, -1, 2)
    # top out of range
    assert_raises(ValueError, integrate_ROI, E, C, 2, 110)
    # different length limits
    assert_raises(ValueError, integrate_ROI, E, C,
                  [32, 1], [2, 10, 32],)
    # independent variable (x_value_array) not increasing monotonically
    assert_raises(ValueError, integrate_ROI, C, C, 2, 10)
    # outliers present in x_value_array which violate monotonic reqirement
    E[2] = 50
    E[50] = 2
    assert_raises(ValueError, integrate_ROI, E, C, 2, 60)

def test_integrate_ROI_compute():
    E = np.arange(100)
    C = np.ones_like(E)
    assert_array_almost_equal(integrate_ROI(E, C, 5.5, 6.5),
                              1)
    assert_array_almost_equal(integrate_ROI(E, C, 5.5, 11.5),
                              6)
    assert_array_almost_equal(integrate_ROI(E, C, [5.5, 17], [11.5, 23]),
                              12)

def test_integrate_ROI_reverse_input():
    E = np.arange(100)
    C = E[::-1]
    E_rev = E[::-1]
    C_rev = C[::-1]
    assert_array_almost_equal(
            integrate_ROI(E_rev, C_rev, [5.5, 17], [11.5, 23]), 
            integrate_ROI(E, C, [5.5, 17], [11.5, 23])
            )
