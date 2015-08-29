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
import logging

import numpy as np
from numpy.testing import (assert_array_almost_equal,
                           assert_almost_equal)
from nose.tools import assert_raises

from skimage import data

import skxray.core.correlation as corr
import skxray.core.roi as roi
import skxray.core.utils as utils


logger = logging.getLogger(__name__)


# It is unclear why this test is so slow. Can we speed this up at all?
def test_correlation():
    num_levels = 4
    num_bufs = 8  # must be even
    num_qs = 2  # number of interested roi's (rings)
    img_dim = (50, 50)  # detector size

    roi_data = np.array(([10, 20, 12, 14], [40, 10, 9, 10]),
                        dtype=np.int64)

    indices = roi.rectangles(roi_data, img_dim)

    img_stack = np.random.randint(1, 5, size=(500, ) + img_dim)

    g2, lag_steps = corr.multi_tau_auto_corr(num_levels, num_bufs, indices,
                                             img_stack)

    assert_array_almost_equal(lag_steps,  np.array([0, 1, 2, 3, 4, 5, 6, 7, 8,
                                                   10, 12, 14, 16, 20, 24, 28,
                                                   32, 40, 48, 56]))

    assert_array_almost_equal(g2[1:, 0], 1.00, decimal=2)
    assert_array_almost_equal(g2[1:, 1], 1.00, decimal=2)


def test_image_stack_correlation():
    num_levels = 1
    num_bufs = 2  # must be even
    coins = data.camera()
    coins_stack = []

    for i in range(4):
        coins_stack.append(coins)

    coins_mesh = np.zeros_like(coins)
    coins_mesh[coins < 30] = 1
    coins_mesh[coins > 50] = 2

    g2, lag_steps = corr.multi_tau_auto_corr(num_levels, num_bufs, coins_mesh,
                                             coins_stack)

    assert_almost_equal(True, np.all(g2[:, 0], axis=0))
    assert_almost_equal(True, np.all(g2[:, 1], axis=0))

    num_buf = 5

    # check the number of buffers are even
    assert_raises(ValueError,
                  lambda: corr.multi_tau_auto_corr(num_levels, num_buf,
                                                   coins_mesh, coins_stack))
    # check image shape and labels shape are equal
    #assert_raises(ValueError,
    #            lambda : corr.multi_tau_auto_corr(num_levels, num_bufs,
    #                                                indices, coins_stack))

    # check the number of pixels is zero
    mesh = np.zeros_like(coins)
    assert_raises(ValueError,
                  lambda: corr.multi_tau_auto_corr(num_levels, num_bufs,
                                                   mesh, coins_stack))


def test_auto_corr_scat_factor():
    num_levels, num_bufs = 3, 4
    tot_channels, lags = utils.multi_tau_lags(num_levels, num_bufs)
    beta = 0.5
    relaxation_rate = 10.0
    baseline = 1.0

    g2 = corr.auto_corr_scat_factor(lags, beta, relaxation_rate, baseline)

    assert_array_almost_equal(g2, np.array([1.5, 1.0, 1.0, 1.0, 1.0,
                                            1.0, 1.0, 1.0]), decimal=8)
