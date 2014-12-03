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
This is a unit/integrated testing script for dpc.py, which conducts
Differential Phase Contrast (DPC) imaging based on Fourier-shift fitting.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)
from nose.tools import (raises, assert_raises)

#import skxray.dpc as dpc
import dpc as dpc

def test_image_reduction_default():
    """
    Test image reduction when default parameters (roi and bad_pixels) are used.

    """

    # Generate a 2D matrix
    img = np.arange(100).reshape(10, 10)

    # Expected results
    xsum = [450, 460, 470, 480, 490, 500, 510, 520, 530, 540]
    ysum = [45, 145, 245, 345, 445, 545, 645, 745, 845, 945]

    # call image reduction
    xline, yline = dpc.image_reduction(img)

    assert_array_equal(xline, xsum)
    assert_array_equal(yline, ysum)

    dpc.image_reduction(img, bad_pixels=[(1, -1), (-1, 1)])
    dpc.image_reduction(img, roi=np.array([0, 0, 20, 20]))


def test_image_reduction():
    """
    Test image reduction when the following parameters are used:
    roi = (3, 3, 5, 5);
    bad_pixels = [(0, 1), (4, 4), (7, 8)]

    """

    # generate a 2D matrix
    img = np.arange(100).reshape(10, 10)

    # set up roi and bad_pixels
    roi = (3, 3, 5, 5)
    bad_pixels = [(0, 1), (4, 4), (7, 8)]

    # Expected results
    xsum = [265, 226, 275, 280, 285]
    ysum = [175, 181, 275, 325, 375]

    # call image reduction
    xline, yline = dpc.image_reduction(img, roi, bad_pixels)

    assert_array_equal(xline, xsum)
    assert_array_equal(yline, ysum)


def test_rss_factory():
    """
    Test _rss_factory.

    """

    length = 10
    v = [2, 3]
    xdata = np.arange(length)
    beta = 1j * (np.arange(length) - length//2)
    ydata = xdata * v[0] * np.exp(v[1] * beta)

    rss = dpc._rss_factory(length)
    residue = rss(v, xdata, ydata)

    assert_almost_equal(residue, 0)
    
    
def test_dpc_fit():
    """
    Test dpc_fit.
    
    """
    
    start_point = [1, 0]
    length = 100
    solver = 'Nelder-Mead'
    xdata = np.arange(length)
    beta = 1j * (np.arange(length) - length//2)
    rss = dpc._rss_factory(length)
    
    # Test 1
    v = [1.02, -0.00023]
    ydata = xdata * v[0] * np.exp(v[1] * beta)
    res = dpc.dpc_fit(rss, xdata, ydata, start_point, solver)
    assert_array_almost_equal(res, v)
    
    # Test 2
    v = [0.88, -0.0048]
    ydata = xdata * v[0] * np.exp(v[1] * beta)
    res = dpc.dpc_fit(rss, xdata, ydata, start_point, solver)
    assert_array_almost_equal(res, v)
    
    # Test 3
    v = [0.98, 0.0068]
    ydata = xdata * v[0] * np.exp(v[1] * beta)
    res = dpc.dpc_fit(rss, xdata, ydata, start_point, solver)
    assert_array_almost_equal(res, v)
    
    # Test 4
    v = [0.95, 0.0032]
    ydata = xdata * v[0] * np.exp(v[1] * beta)
    res = dpc.dpc_fit(rss, xdata, ydata, start_point, solver)
    assert_array_almost_equal(res, v)
    
    
def test_dpc_end_to_end():
    """
    Integrated test for DPC based on dpc_runner.
    
    """
    
    start_point = [1, 0]
    pixel_size = (55, 55)
    focus_to_det = 1.46e6
    rows = 2
    cols = 2
    dx = 0.1
    dy = 0.1
    energy = 19.5
    roi = None
    padding = 0
    w = 1
    bad_pixels = None
    solver = 'Nelder-Mead'
    img_size = (40, 40)
    scale = True

    ref_image = np.ones(img_size)
    image_sequence = np.ones((rows * cols, img_size[0], img_size[1]))

    phi = dpc.dpc_runner(start_point, pixel_size, focus_to_det, rows, cols, dx,
                         dy, energy, roi, bad_pixels, ref_image, 
                         image_sequence, solver, padding, w, scale)

    assert_array_almost_equal(phi, np.zeros((rows, cols)))


if __name__ == "__main__":

    test_image_reduction_default()
    test_image_reduction()
    test_rss_factory()
    test_dpc_fit()

    test_dpc_end_to_end()
