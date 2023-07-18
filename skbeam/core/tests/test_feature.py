# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# @author: Li Li (lili@bnl.gov)                                        #
# created on 08/19/2014                                                #
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
import pytest
from numpy.testing import assert_array_almost_equal, assert_raises

import skbeam.core.feature as feature

from .utils import gauss_gen, parabola_gen


def _test_refine_helper(x_data, y_data, center, height, refine_method, refine_args):
    """
    helper function for testing
    """
    test_center, test_height = refine_method(x_data, y_data, **refine_args)
    assert_array_almost_equal(np.array([test_center, test_height]), np.array([center, height]))


def _gen_test_refine_methods():
    refine_methods = [feature.refine_quadratic, feature.refine_log_quadratic]
    test_data_gens = [parabola_gen, gauss_gen]

    x = np.arange(128)

    vals = []
    for center in (15, 75, 110):
        for height in (5, 10, 100):
            for rf, dm in zip(refine_methods, test_data_gens):
                vals.append((x, dm(x, center, height, 5), center, height, rf))

    return vals


param_test_refine_methods = _gen_test_refine_methods()


@pytest.mark.parametrize("x, dm, center, height, rf", param_test_refine_methods)
def test_refine_methods(x, dm, center, height, rf):
    _test_refine_helper(x, dm, center, height, rf, {})


def test_filter_n_largest():
    cands = np.array((10, 25, 50, 75, 100))
    x = np.arange(128, dtype=float)
    y = np.zeros_like(x)
    for c, h in zip(cands, (10, 15, 25, 30, 35)):
        y += gauss_gen(x, c, h, 3)

    for j in range(1, len(cands) + 2):
        out = feature.filter_n_largest(y, cands, j)
        assert len(out) == np.min([len(cands), j])

    assert_raises(ValueError, feature.filter_n_largest, y, cands, 0)
    assert_raises(ValueError, feature.filter_n_largest, y, cands, -1)


def test_filter_peak_height():
    cands = np.array((10, 25, 50, 75, 100))
    heights = (10, 20, 30, 40, 50)
    x = np.arange(128, dtype=float)
    y = np.zeros_like(x)
    for c, h in zip(cands, heights):
        y += gauss_gen(x, c, h, 3)

    for j, h in enumerate(heights):
        out = feature.filter_peak_height(y, cands, h - 5, window=5)
        assert len(out) == len(heights) - j
        out = feature.filter_peak_height(y, cands, h + 5, window=5)
        assert len(out) == len(heights) - j - 1


def test_peak_refinement():
    cands = np.array((10, 25, 50, 75, 100))
    heights = (10, 20, 30, 40, 50)
    x = np.arange(128, dtype=float)
    y = np.zeros_like(x)
    for c, h in zip(cands, heights):
        y += gauss_gen(x, c + 0.5, h, 3)

    loc, ht = feature.peak_refinement(x, y, cands, 5, feature.refine_log_quadratic)
    assert_array_almost_equal(loc, cands + 0.5, decimal=3)
    assert_array_almost_equal(ht, heights, decimal=3)
