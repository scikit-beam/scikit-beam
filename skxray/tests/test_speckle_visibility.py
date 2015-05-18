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

from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)
import sys

from nose.tools import assert_equal, assert_true, assert_raises

import skxray.correlation as corr
import skxray.roi as roi
import skxray.speckle_visibility as xsvs

from skxray.testing.decorators import known_fail_if
import numpy.testing as npt

from skimage import data, morphology


def test_intensity_distribution():
    image_array = data.moon()
    # width incompatible with num_rings

    label_array = np.zeros((256, 256))

    # different shapes for the images and labels
    assert_raises(ValueError,
            lambda: xsvs.intensity_distribution(image_array,
                                                label_array))

    images = morphology.diamond(8)

    # create a label mask
    center = (8., 8.)
    inner_radius = 2.
    width = 1
    spacing = 1
    edges = roi.ring_edges(inner_radius, width, spacing, num_rings=5)
    rings = roi.rings(edges, center, images.shape)

    intensity_dist = xsvs.intensity_distribution(images, rings)
    assert_array_equal(intensity_dist.values()[0], ([1, 1, 1, 1, 1, 1, 1, 1,
                                                     1, 1, 1, 1, 1, 1, 1, 1]))

def test_static_test_sets():
    image_dict =
    label_array =
    num = 1


def test_static_tests_one_label():
    images =  data.moon()
    label_array = np.zeros((256, 256))

    # different shapes for the images and labels
    assert_raises(ValueError,
                  lambda: xsvs.static_tests_one_label(images, label_array))


def static_test():
    images =
    label_array =


def test_time_bining():
    time_bin = xsvs.time_bining(number=5, number_of_images=150)

    assert_array_equal(time_bin, [1, 5, 25, 125])


def test_max_counts():
    img_stack1 = np.random.randint(0, 50, size=(50, ) + (50, 50))
    img_stack2 = np.random.randint(0, 50, size=(100, ) + (50, 50))

    img_stack1[0][50, 50] = 60

    sample_dict = {1:np.nditer(img_stack1), 2:np.nditer(img_stack2)}

    label_array = np.zeros((img_stack1[0].shape))

    label_array[img_stack1[0] < 20] = 1
    label_array[img_stack1[0] > 40] = 2

    assert_array_equal(60, xsvs.max_counts(sample_dict, label_array))