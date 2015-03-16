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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
import logging
logger = logging.getLogger(__name__)
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)
import sys

from nose.tools import assert_equal, assert_true, raises

import skxray.correlation as corr
import skxray.diff_roi_choice as diff_roi

from skxray.testing.decorators import known_fail_if
import numpy.testing as npt


def test_correlation():
    num_levels = 4
    num_bufs = 4  # must be even
    num_qs = 2  # number of interested roi's (rings)
    img_dim = (150, 150) # detector size
    calibrated_center = (60., 55.)
    first_r = 20.  # radius of the first ring
    delta_r = 10.  # thickness of the rings

    (q_inds, ring_vals, num_pixels,
     pixel_list) = diff_roi.roi_rings(img_dim, calibrated_center,
                                      num_qs, first_r, delta_r)
    roi_data = np.array(([60, 70, 12, 6], [140, 120, 5, 10]),
                                       dtype=np.int64)

    (q_inds, num_pixels,
     pixel_list) = diff_roi.roi_rectangles(num_qs, roi_data, img_dim)

    img_stack = []
    for i in range(500):
        img_stack.append(np.random.randint(1, 5, size=(img_dim)))

    g2, lag_steps, elapsed_time = corr.auto_corr(num_levels, num_bufs,
                                                 num_qs, num_pixels,
                                                 pixel_list, q_inds,
                                                 np.asarray(img_stack))

    assert_array_almost_equal(lag_steps, np.array([0, 1, 2, 3, 4, 6, 8, 12, 16, 24]))

    g2_m = np. array([[1.200, 1.200],
                      [0.998, 1.000],
                      [0.998, 0.997],
                      [0.999, 1.000],
                      [1.000, 1.000],
                      [1.000, 1.000],
                      [0.999, 1.000],
                      [0.999, 1.000],
                      [1.000, 1.000],
                      [0.999, 1.000]])

    assert_array_almost_equal(g2, g2_m, decimal=2)
