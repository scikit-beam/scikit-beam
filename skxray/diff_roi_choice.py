#! encoding: utf-8
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
This module is to get informations of different region of interests(roi's).
Information : the number of pixels, pixel indices, indices
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six

from six.moves import zip
from six import string_types

import time
import sys
import numpy as np
import numpy.ma as ma

import skxray.correlation as corr
import skxray.core as core

import logging
logger = logging.getLogger(__name__)


def rectangles(num_rois, roi_data, image_shape):
    """
    Parameters
    ----------
    num_rois: int
        number of region of interests(roi)

    roi_data: ndarray
        upper left co-ordinates of roi's and the, length and width of roi's
        from those co-ordinates
        shape is [num_rois][4]

    image_shape : tuple
        2 element tuple defining the number of pixels in the detector.
        Order is (num_rows, num_columns)

    Returns
    -------
    labels_grid : array
        indices of the required rings
        shape is ([image_shape[0]*image_shape[1]], )

    """

    labels_grid = np.zeros(image_shape, dtype=np.int64)

    for i, (col_coor, row_coor, col_val, row_val) in enumerate(roi_data, 0):

        left, right = np.max([col_coor, 0]), np.min([col_coor + col_val,
                                                     image_shape[0]])
        top, bottom = np.max([row_coor, 0]), np.min([row_coor + row_val,
                                                     image_shape[1]])

        area = (right - left) * (bottom - top)

        slc1 = slice(left, right)
        slc2 = slice(top, bottom)

        if np.any(labels_grid[slc1, slc2]):
            raise ValueError("overlapping ROIs")

        # assign a different scalar for each roi
        labels_grid[slc1, slc2] = (i + 1)

    return labels_grid


def rings(image_shape, calibrated_center, num_rings,
          first_r, delta_r):
    """
    This will provide the indices of the required rings,
    find the bin edges of the rings, and count the number
    of pixels in each ring, and pixels indices for the
    required rings when there is no step value between
    rings.

    Parameters
    ----------
    image_shape: tuple
        shape of the image (detector X and Y direction)
        Order is (num_rows, num_columns)
        shape is [image_shape[0], image_shape[1]])

    calibrated_center : tuple
        defining the center of the image
        (column value, row value) (mm)

    num_rings : int
        number of rings

    first_r : float
        radius of the first  ring

    delta_r : float
        thickness of the ring

    Returns
    -------
    labels_grid : array
        indices of the required rings
        shape is ([image_shape[0]*image_shape[1]], )

    """
    grid_values = core.pixel_to_radius(image_shape, calibrated_center)

    edges = rings_edges(num_rings, first_r, delta_r)

    # indices of rings
    labels_grid = np.digitize(np.ravel(grid_values), np.array(edges),
                              right=False)
    # discard the indices greater than number of rings
    labels_grid[labels_grid > num_rings] = 0

    return labels_grid.reshape(image_shape)


def rings_edges(num_rings, first_r, delta_r):
    """
    This function will provide the edge values of the rings when
    there is a no step value between each ring

    Parameters
    ----------
    num_rings : int
        number of rings

    first_r : float
        radius of the first  ring

    delta_r : float
        thickness of the ring

    Returns
    -------
    ring_vals : array
        edge values of each ring

    ring_edges : array
        edge values of each ring
        shape is (num_rings, 2)
    """

    # last ring edge value
    last_r = first_r + num_rings*(delta_r)

    # edges of all the rings
    ring_edges = np.linspace(first_r, last_r, num=(num_rings+1))

    return ring_edges


def rings_step(image_shape, calibrated_center, num_rings, first_r, delta_r,
               *step_r):
    """
    This will provide the indices of the required rings,
    find the bin edges of the rings, and count the number
    of pixels in each ring, and pixels indices for the
    required rings when there is a step value between rings.
    Step value can be same or different steps between
    each ring.

    num_rings : int
        number of rings

    first_r : float
        radius of the first ring

    delta_r : float
        thickness of the ring

    calibarted_center : tuple
        defining the center of the image (column value, row value) (mm)

    step_r : tuple
        step value for the next ring from the end of the previous
        ring.
        same step - same step values between rings (one value)
        different steps - different step value between rings (provide
        step value for each ring eg: 6 rings provide 5 step values)

    Returns
    -------
    labels_grid : array
        indices of the required rings
        shape is ([image_shape[0]*image_shape[1]], )
    """
    grid_values = core.pixel_to_radius(image_shape, calibrated_center)

    ring_vals = rings_step_edges(num_rings, first_r, delta_r, *step_r)

    # indices of rings
    labels_grid = np.digitize(np.ravel(grid_values), np.array(ring_vals),
                              right=False)

    # to discard every-other bin and set the discarded bins indices to 0
    labels_grid[labels_grid % 2 == 0] = 0
    # change the indices of odd number of rings
    indx = labels_grid > 0
    labels_grid[indx] = (labels_grid[indx] + 1) // 2

    return labels_grid.reshape(image_shape)


def rings_step_edges(num_rings, first_r, delta_r, *step_r):
    """
    This function will provide the edge values of the rings when there is
    a step value between each ring

    Parameters
    ----------
    num_rings : int
        number of rings

    first_r : float
        radius of the first ring

    delta_r : float
        thickness of the ring

    calibarted_center : tuple
        defining the center of the image (column value, row value) (mm)

    step_r : tuple
        step value for the next ring from the end of the previous
        ring.
        same step - same step values between rings (one value)
        different steps - different step value between rings (provide
        step value for each ring eg: 6 rings provide 5 step values)

    Returns
    -------
    ring_vals : array
        edge values of each ring
    """
    ring_vals = []

    for arg in step_r:
        if arg < 0:
            raise ValueError("step value for the next ring from the "
                             "end of the previous ring has to be positive ")

    if len(step_r) == 1:
        #  when there is a same values of step between rings
        #  the edge values of rings will be
        ring_vals = first_r + np.r_[0, np.cumsum(np.tile([delta_r,
                                                          float(step_r[0])],
                                                         num_rings))][:-1]
    else:
        # when there is a different step values between each ring
        #  edge values of the rings will be
        if len(step_r) == (num_rings - 1):
            ring_vals.append(first_r)
            for arg in step_r:
                ring_vals.append(ring_vals[-1] + delta_r)
                ring_vals.append(ring_vals[-1] + float(arg))
            ring_vals.append(ring_vals[-1] + delta_r)
        else:
            raise ValueError("Provide step value for each q ring ")

    return ring_vals


def process_ring_edges(ring_vals):
    """
    This function will provide edge values
    of the each roi ring shape as (num_rings, 2)

    Parameters
    ----------
    ring_vals : array
        edge values of each ring

    Returns
    -------
    ring_vals : array
        edge values of each ring
        shape is (num_rings, 2)

    """
    ring_vals = np.asarray(ring_vals).reshape(-1, 2)

    return ring_vals
