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

from skimage.morphology import convex_hull_image

import skxray.core.correlation as corr
import skxray.core.xsvs as xsvs
import skxray.core.xsvs_fitting as xsvs_fit
import skxray.core.roi as roi
from skxray.testing.decorators import skip_if

logger = logging.getLogger(__name__)


def test_xsvs():
    images = []
    for i in range(10):
        int_array = np.tril(i*np.ones(10))
        if i == 10/2:
            int_array[int_array == 0] = 20
        else:
            int_array[int_array == 0] = i*2
        images.append(int_array)

    images = np.asarray(images)
    roi_data = np.array(([4, 2, 2, 2], [0, 5, 2, 2]), dtype=np.int64)
    label_array = roi.rectangles(roi_data, shape=images[0].shape)

    num_times = 4
    num_rois = 2


def test_normalize_bin_edges():
    num_times = 3
    num_rois = 2
    mean_roi = np.array([2.5, 4.0])
    max_cts = 5

    bin_edges, bin_cen = xsvs.normalize_bin_edges(num_times, num_rois,
                                                  mean_roi, max_cts)

    assert_array_almost_equal(bin_edges[0, 0], np.array([0., 0.4, 0.8,
                                                        1.2, 1.6]))

    assert_array_almost_equal(bin_edges[2, 1], np.array([0., 0.0625, 0.125,
                                                         0.1875, 0.25, 0.3125,
                                                         0.375, 0.4375, 0.5,
                                                         0.5625, 0.625, 0.6875,
                                                         0.75, 0.8125, 0.875,
                                                         0.9375, 1., 1.0625,
                                                         1.125, 1.1875]))

    assert_array_almost_equal(bin_cen[0, 0], np.array([0.2, 0.6, 1., 1.4]))


def test_distribution():
    M = 1.9  # number of coherent modes
    K = 3.15  # number of photons

    bin_edges = np.array([0., 0.4, 0.8, 1.2, 1.6, 2.0])

    p_k = xsvs_fit.negative_binom_distribution(bin_edges, K, M)

    poission_dist = xsvs_fit.poisson_distribution(bin_edges, K)

    gamma_dist = xsvs_fit.gamma_distribution(bin_edges, M, K)

    assert_array_almost_equal(p_k, np.array([0.15609113, 0.17669628,
                                             0.18451672, 0.1837303,
                                             0.17729389, 0.16731627]))
    assert_array_almost_equal(gamma_dist, np.array([0., 0.13703903, 0.20090424,
                                                    0.22734693, 0.23139384,
                                                    0.22222281]))
    assert_array_almost_equal(poission_dist, np.array([0.04285213, 0.07642648,
                                                       0.11521053, 0.15411372,
                                                       0.18795214, 0.21260011]))


def test_diffusive_motion_contrast_factor():
    times = np.array([1, 2, 4, 8])
    relaxation_rate = 6.40
    contrast_factor = 0.17
    cf_baseline = 0

    diff_con_fac = xsvs_fit.diffusive_motion_contrast_factor(times,
                                                             relaxation_rate,
                                                             contrast_factor,
                                                             cf_baseline)
    assert_array_almost_equal(diff_con_fac, np.array([0.02448731, 0.01276245,
                                                      0.00651093, 0.00328789]))


def test_diffusive_coefficient():
    relaxation_rates = np.array([6.40, 6.80, 7.30, 7.80])
    q_values = np.array([0.0026859, 0.00278726, 0.00288861, 0.00298997])

    diff_co = xsvs_fit.diffusive_coefficient(relaxation_rates, q_values)

    assert_array_almost_equal(diff_co, np.array([887156.61579, 875293.9933,
                                                 874873.0516, 872490.9704]),
                              decimal=4)
