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
from __future__ import absolute_import, division, print_function

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_raises

from skbeam.core.spectroscopy import align_and_scale, integrate_ROI, integrate_ROI_spectrum


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
    return alpha * np.exp(-((E - E0) ** 2) / (2 * sigma**2)) + beta * (1 + np.sin(k * E))


def test_align_and_scale_smoketest():
    # does nothing but call the function

    # make data
    E = np.linspace(0, 50, 1000)
    # this is not efficient for large lists, but quick and dirty
    e_list = []
    c_list = []
    for j in range(25, 35, 2):
        e_list.append(E)
        c_list.append(synthetic_data(E, j + j / 100, j / 10, 1000, 2 * np.pi * 6 / 50, 60))
    # call the function
    e_cor_list, c_cor_list = align_and_scale(e_list, c_list)


def test_integrate_ROI_errors():
    E = np.arange(100)
    C = np.ones_like(E)

    # limits out of order
    assert_raises(ValueError, integrate_ROI, E, C, [32, 1], [2, 10])
    # bottom out of range
    assert_raises(ValueError, integrate_ROI, E, C, -1, 2)
    # top out of range
    assert_raises(ValueError, integrate_ROI, E, C, 2, 110)
    # different length limits
    assert_raises(
        ValueError,
        integrate_ROI,
        E,
        C,
        [32, 1],
        [2, 10, 32],
    )
    # independent variable (x_value_array) not increasing monotonically
    assert_raises(ValueError, integrate_ROI, C, C, 2, 10)
    # outliers present in x_value_array which violate monotonic reqirement
    E[2] = 50
    E[50] = 2
    assert_raises(ValueError, integrate_ROI, E, C, 2, 60)


def test_integrate_ROI_compute():
    E = np.arange(100)
    C = np.ones_like(E)
    assert_array_almost_equal(integrate_ROI(E, C, 5.5, 6.5), 1)
    assert_array_almost_equal(integrate_ROI(E, C, 5.5, 11.5), 6)
    assert_array_almost_equal(integrate_ROI(E, C, [5.5, 17], [11.5, 23]), 12)


def test_integrate_ROI_spectrum_compute():
    C = np.ones(100)
    E = np.arange(101)
    assert_array_almost_equal(integrate_ROI_spectrum(E, C, 5, 6), 1)
    assert_array_almost_equal(integrate_ROI_spectrum(E, C, 5, 11), 6)
    assert_array_almost_equal(integrate_ROI_spectrum(E, C, [5, 17], [11, 23]), 12)


def test_integrate_ROI_reverse_input():
    E = np.arange(100)
    C = E[::-1]
    E_rev = E[::-1]
    C_rev = C[::-1]
    assert_array_almost_equal(
        integrate_ROI(E_rev, C_rev, [5.5, 17], [11.5, 23]), integrate_ROI(E, C, [5.5, 17], [11.5, 23])
    )
