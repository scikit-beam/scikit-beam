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

import skxray.core.speckle_fitting as xsvs_fit

from skxray.testing.decorators import skip_if

logger = logging.getLogger(__name__)


def test_distribution():
    M = 1.9  # number of coherent modes
    K = 3.15  # number of photons

    bin_edges = np.array([0., 0.4, 0.8, 1.2, 1.6, 2.0])

    p_k = xsvs_fit.negative_binom_distribution(bin_edges, K, M)

    poission_dist = xsvs_fit.poisson_distribution(bin_edges, K)

    gamma_dist = xsvs_fit.gamma_distribution(bin_edges, K, M)

    assert_array_almost_equal(p_k, np.array([0.15609113, 0.17669628,
                                             0.18451672, 0.1837303,
                                             0.17729389, 0.16731627]))
    assert_array_almost_equal(gamma_dist, np.array([0., 0.13703903, 0.20090424,
                                                    0.22734693, 0.23139384,
                                                    0.22222281]))
    assert_array_almost_equal(poission_dist, np.array([0.04285213, 0.07642648,
                                                       0.11521053, 0.15411372,
                                                       0.18795214, 0.21260011]))
