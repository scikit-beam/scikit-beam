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
from numpy.testing import assert_array_equal

import skbeam.core.mask as mask

logger = logging.getLogger(__name__)


def test_threshold_mask():
    xdim = 10
    ydim = 10
    stack_size = 10
    img_stack = np.random.randint(1, 3, (stack_size, xdim, ydim))

    img_stack[0][0, 1] = 100
    img_stack[0][9, 1] = 98
    img_stack[6][8, 8] = 75
    img_stack[7][6, 6] = 80

    th = mask.threshold(img_stack, 75)

    for final in th:
        pass

    y = np.ones_like(img_stack[0])
    y[0, 1] = 0
    y[9, 1] = 0
    y[8, 8] = 0
    y[6, 6] = 0

    assert_array_equal(final, y)


def test_bad_to_nan_gen():
    xdim = 2
    ydim = 2
    stack_size = 5
    img_stack = np.random.randint(1, 3, (stack_size, xdim, ydim))

    bad_list = [1, 3]

    img = mask.bad_to_nan_gen(img_stack, bad_list)
    y = []
    for im in img:
        y.append(im)

    assert np.isnan(np.asarray(y)[1]).all()
    assert np.isnan(np.asarray(y)[3]).all()
    assert not np.isnan(np.asarray(y)[4]).all()


def test_margin():
    size = (10, 10)
    edge = 1
    mask1 = mask.margin(size, edge)
    mask2 = np.zeros(size)
    mask2[:, :edge] = 1
    mask2[:, -edge:] = 1
    mask2[:edge, :] = 1
    mask2[-edge:, :] = 1
    mask2 = mask2.astype(bool)
    assert_array_equal(mask1, ~mask2)


def test_ring_blur_mask():
    from skbeam.core import recip

    g = recip.geo.Geometry(
        detector="Perkin",
        pixel1=0.0002,
        pixel2=0.0002,
        dist=0.23,
        poni1=0.209,
        poni2=0.207,
        # rot1=.0128, rot2=-.015, rot3=-5.2e-8,
        wavelength=1.43e-11,
    )
    r = g.rArray((2048, 2048))
    # make some sample data
    Z = 100 * np.cos(50 * r) ** 2 + 150

    np.random.seed(10)
    pixels = []
    for i in range(0, 100):
        a, b = np.random.randint(low=0, high=2048), np.random.randint(low=0, high=2048)
        if np.random.random() > 0.5:
            # Add some hot pixels
            Z[a, b] = np.random.randint(low=200, high=255)
        else:
            # and dead pixels
            Z[a, b] = np.random.randint(low=0, high=10)
        pixels.append((a, b))
    pixel_size = [getattr(g, k) for k in ["pixel1", "pixel2"]]
    rres = np.hypot(*pixel_size)
    bins = np.arange(np.min(r) - rres / 2.0, np.max(r) + rres / 2.0, rres)
    msk = mask.binned_outlier(Z, r, (3.0, 3), bins, mask=None)
    a = set(zip(*np.nonzero(~msk)))
    b = set(pixels)
    a_not_in_b = a - b
    b_not_in_a = b - a

    # We have not over masked 10% of the number of bad pixels
    assert len(a_not_in_b) / len(b) < 0.1
    # Make certain that we have masked over 90% of the bad pixels
    assert len(b_not_in_a) / len(b) < 0.1
