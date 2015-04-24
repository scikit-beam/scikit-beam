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

from nose.tools import assert_equal, assert_true, assert_raises

import skxray.roi as roi
import skxray.correlation as corr
import skxray.core as core

from skxray.testing.decorators import known_fail_if
import numpy.testing as npt


def test_rectangles():
    shape = (15, 26)
    roi_data = np.array(([2, 2, 6, 3], [6, 7, 8, 5], [8, 18, 5, 10]),
                        dtype=np.int64)

    all_roi_inds = roi.rectangles(roi_data, shape)

    roi_inds, pixel_list = corr.extract_label_indices(all_roi_inds)

    ty = np.zeros(shape).ravel()
    ty[pixel_list] = roi_inds
    num_pixels_m = (np.bincount(ty.astype(int)))[1:]

    re_mesh = ty.reshape(*shape)
    for i, (col_coor, row_coor, col_val, row_val) in enumerate(roi_data, 0):
        ind_co = np.column_stack(np.where(re_mesh == i + 1))

        left, right = np.max([col_coor, 0]), np.min([col_coor + col_val,
                                                     shape[0]])
        top, bottom = np.max([row_coor, 0]), np.min([row_coor + row_val,
                                                     shape[1]])
        assert_almost_equal(left, ind_co[0][0])
        assert_almost_equal(right-1, ind_co[-1][0])
        assert_almost_equal(top, ind_co[0][1])
        assert_almost_equal(bottom-1, ind_co[-1][-1])


def test_rings():
    calibrated_center = (100., 100.)
    img_dim = (200, 205)
    first_q = 10.
    delta_q = 5.
    num_rings = 7  # number of Q rings
    one_step_q = 5.0
    step_q = [2.5, 3.0, 5.8]

    # test when there is same spacing between rings
    edges = roi.ring_edges(first_q, width=delta_q, spacing=one_step_q,
                           num_rings=num_rings)
    print("edges there is same spacing between rings ", edges)
    label_array = roi.rings(edges, calibrated_center, img_dim)
    print("label_array there is same spacing between rings", label_array)

    # test when there is same spacing between rings
    edges = roi.ring_edges(first_q, width=delta_q, spacing=2.5,
                           num_rings=num_rings)
    print("edges there is same spacing between rings ", edges)
    label_array = roi.rings(edges, calibrated_center, img_dim)
    print("label_array there is same spacing between rings", label_array)

    # test when there is different spacing between rings
    edges = roi.ring_edges(first_q, width=delta_q, spacing=step_q,
                           num_rings=4)
    print("edges when there is different spacing between rings", edges)
    label_array = roi.rings(edges, calibrated_center, img_dim)
    print("label_array there is different spacing between rings", label_array)

    # test when there is no spacing between rings
    edges = roi.ring_edges(first_q, width=delta_q, num_rings=num_rings)
    print("edges", edges)
    label_array = roi.rings(edges, calibrated_center, img_dim)
    print("label_array", label_array)

    # Did we draw the right number of rings?
    print(np.unique(label_array))
    actual_num_rings = len(np.unique(label_array)) - 1
    assert_equal(actual_num_rings, num_rings)

    # Does each ring have more pixels than the last, being larger?
    ring_areas = np.bincount(label_array.ravel())[1:]
    area_comparison = np.diff(ring_areas)
    print(area_comparison)
    areas_monotonically_increasing = np.all(area_comparison > 0)
    assert_true(areas_monotonically_increasing)

    # Test various illegal inputs
    assert_raises(ValueError,
            lambda: roi.ring_edges(1, 2))  # need num_rings
    # width incompatible with num_rings
    assert_raises(ValueError,
            lambda: roi.ring_edges(1, [1, 2, 3], num_rings=2))
    # too few spacings
    assert_raises(ValueError,
            lambda: roi.ring_edges(1, [1, 2, 3], [1]))
    # too many spacings
    assert_raises(ValueError,
            lambda: roi.ring_edges(1, [1, 2, 3], [1, 2, 3]))
    # num_rings conflicts with width, spacing
    assert_raises(ValueError,
            lambda: roi.ring_edges(1, [1, 2, 3], [1, 2], 5))


def _helper_check(pixel_list, inds, num_pix, q_ring_val, calib_center,
                  img_dim, num_qs):
    # recreate the indices using pixel_list and inds values
    ty = np.zeros(img_dim).ravel()
    ty[pixel_list] = inds
    data = ty.reshape(img_dim[0], img_dim[1])

    # get the grid values from the center
    grid_values = core.pixel_to_radius(img_dim, calib_center)

    # get the indices into a grid
    zero_grid = np.zeros((img_dim[0], img_dim[1]))
    for r in range(num_qs):
        vl = (q_ring_val[r][0] <= grid_values) & (grid_values
                                                  < q_ring_val[r][1])
        zero_grid[vl] = r + 1

    # check the num_pixels
    num_pixels = []
    for r in range(num_qs):
        num_pixels.append(int((np.histogramdd(np.ravel(grid_values), bins=1,
                                              range=[[q_ring_val[r][0],
                                                      (q_ring_val[r][1]
                                                       - 0.000001)]]))[0][0]))
    assert_array_equal(zero_grid, data)
    assert_array_equal(num_pix, num_pixels)


def test_divide_pies():
    detector_size = (100, 80)
    radius = 50.
    calibrated_center = (50., 20.)
    num_angles = 8

    angles = np.linspace(0, 360, num_angles)

    all_roi_inds = roi.divide_pies(detector_size, radius,
                                         calibrated_center,
                                         num_angles)

    labels, indices =  corr.extract_label_indices(all_roi_inds)

    ty = np.zeros(detector_size).ravel()
    ty[indices] = labels

    num_pixels = (np.bincount(ty.astype(int)))[1:]

    # get the angle grid from the center
    angle_grid = roi.get_angle_grid(detector_size, calibrated_center)
    # get the indices into a grid
    zero_grid = np.zeros((detector_size[0], detector_size[1]))

    # radius values from the calibrated
    grid_values = core.pixel_to_radius(detector_size, calibrated_center)

    zero_grid[grid_values <= radius] = 1

    assert_array_equal(np.nonzero(zero_grid),
                       np.nonzero((ty.reshape(*detector_size))))

    mesh = np.zeros((detector_size[0], detector_size[1]))
    for i in range(num_angles-1):
        vl = ((angles[i] <= angle_grid) &
              (angle_grid < angles[i + 1]))
        mesh[vl] = i + 1

    # remove the values grater than the radius
    mesh[grid_values > radius] = 0
    # take out the zero values in the grid
    roi_inds_m = mesh[mesh > 0]

    assert_array_equal(roi_inds_m, labels)
