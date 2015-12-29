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
from numpy.testing import (assert_array_almost_equal, assert_array_equal)
from nose.tools import assert_raises

from skimage import data

import skxray.core.correlation as corr
import skxray.core.roi as roi
import skxray.core.utils as utils


logger = logging.getLogger(__name__)


class FakeStack:
    """Fake up a big pile of images that are identical
    """
    def __init__(self, ref_img, maxlen):
        """

        Parameters
        ----------
        ref_img : array
            The reference image that will be returned `maxlen` times
        maxlen : int
            The maximum number of images to fake up
        """
        self.img = ref_img
        self.maxlen = maxlen

    def __len__(self):
        return self.maxlen

    def __getitem__(self, item):
        if item > len(self):
            raise IndexError
        return self.img


# It is unclear why this test is so slow. Can we speed this up at all?
def test_correlation():
    num_levels = 4
    num_bufs = 8  # must be even
    img_dim = (50, 50)  # detector size

    roi_data = np.array(([10, 20, 12, 14], [40, 10, 9, 10]),
                        dtype=np.int64)

    indices = roi.rectangles(roi_data, img_dim)

    img_stack = np.random.randint(1, 5, size=(64, ) + img_dim)

    g2, lag_steps = corr.multi_tau_auto_corr(num_levels, num_bufs, indices,
                                             img_stack)

    assert_array_equal(lag_steps,  np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10,
                                             12, 14, 16, 20, 24, 28, 32, 40,
                                             48, 56]))

    assert_array_almost_equal(g2[1:, 0], 1.00, decimal=2)
    assert_array_almost_equal(g2[1:, 1], 1.00, decimal=2)


def test_image_stack_correlation():
    num_levels = 4
    num_bufs = 4  # must be even
    xdim = 256
    ydim = 512
    img_stack = FakeStack(ref_img=np.zeros((xdim, ydim), dtype=int), maxlen=20)

    rois = np.zeros_like(img_stack[0])
    # make sure that the ROIs can be any integers greater than 1. They do not
    # have to start at 1 and be continuous
    rois[0:xdim//10, 0:ydim//10] = 5
    rois[xdim//10:xdim//5, ydim//10:ydim//5] = 3

    g2, lag_steps = corr.multi_tau_auto_corr(num_levels, num_bufs, rois,
                                             img_stack)

    assert np.all(g2[:, 0], axis=0)
    assert np.all(g2[:, 1], axis=0)

    # Make sure that an odd number of buffers raises a Value Error
    num_buf = 5
    assert_raises(ValueError, corr.multi_tau_auto_corr, num_levels, num_buf,
                  rois, img_stack)

    # If there are no ROIs, g2 should be an empty array
    rois = np.zeros_like(img_stack[0])
    g2, lag_steps = corr.multi_tau_auto_corr(num_levels, num_bufs, rois,
                                             img_stack)
    assert np.all(g2 == [])


def test_auto_corr_scat_factor():
    num_levels, num_bufs = 3, 4
    tot_channels, lags = utils.multi_tau_lags(num_levels, num_bufs)
    beta = 0.5
    relaxation_rate = 10.0
    baseline = 1.0

    g2 = corr.auto_corr_scat_factor(lags, beta, relaxation_rate, baseline)

    assert_array_almost_equal(g2, np.array([1.5, 1.0, 1.0, 1.0, 1.0,
                                            1.0, 1.0, 1.0]), decimal=8)

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
