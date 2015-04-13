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

import logging
logger = logging.getLogger(__name__)


def roi_rectangles(num_rois, roi_data, detector_size):
    """
    Parameters
    ----------
    num_rois: int
        number of region of interests(roi)

    roi_data: ndarray
        upper left co-ordinates of roi's and the, length and width of roi's
        from those co-ordinates
        shape is [num_rois][4]

    detector_size : tuple
        2 element tuple defining the number of pixels in the detector.
        Order is (num_rows, num_columns)

    mask : ndarray, optional
        masked array
        shape is ([detector_size[0], detector_size[1]])


    Returns
    -------
    roi_inds : ndarray
        indices of the required shape
         after discarding zero values from the shape
        ([detector_size[0]*detector_size[1]], )

    num_pixels : ndarray
        number of pixels in certain rectangle shape

    pixel_list : ndarray
        pixel indices for the required rectangles
    """

    mesh = np.zeros(detector_size, dtype=np.int64)

    num_pixels = []

    for i, (col_coor, row_coor, col_val, row_val) in enumerate(roi_data, 0):

        left, right = np.max([col_coor, 0]), np.min([col_coor + col_val,
                                                     detector_size[0]])
        top, bottom = np.max([row_coor, 0]), np.min([row_coor + row_val,
                                                     detector_size[1]])

        area = (right - left) * (bottom - top)

        # find the number of pixels in each roi
        num_pixels.append(area)

        slc1 = slice(left, right)
        slc2 = slice(top, bottom)

        if np.any(mesh[slc1, slc2]):
            raise ValueError("overlapping ROIs")

        # assign a different scalar for each roi
        mesh[slc1, slc2] = (i + 1)

    # find the pixel indices
    w = np.where(mesh > 0)
    grid = np.indices((detector_size[0], detector_size[1]))
    pixel_list = ((grid[0]*detector_size[1] + grid[1]))[w]

    roi_inds = mesh[mesh > 0]

    return mesh.reshape(detector_size), roi_inds, num_pixels, pixel_list


def roi_rings(img_dim, calibrated_center, num_rings,
               first_r, delta_r):
    """
    This will provide the indices of the required rings,
    find the bin edges of the rings, and count the number
    of pixels in each ring, and pixels indices for the
    required rings when there is no step value between
    rings.

    Parameters
    ----------
    img_dim: tuple
        shape of the image (detector X and Y direction)
        Order is (num_rows, num_columns)
        shape is [detector_size[0], detector_size[1]])

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
    mesh : nedarry
        indices of the required rings
        shape is ([detector_size[0]*detector_size[1]], )

    ring_inds : ndarray
        indices of the required rings

    ring_vals : ndarray
        edge values of the required  rings

    num_pixels : ndarray
        number of pixels in each ring

    pixel_list : ndarray
        pixel indices for the required rings

    """
    grid_values = _grid_values(img_dim, calibrated_center)

    # last ring edge value
    last_r = first_r + num_rings*(delta_r)

    # edges of all the rings
    q_r = np.linspace(first_r, last_r, num=(num_rings+1))

    # indices of rings
    mesh = np.digitize(np.ravel(grid_values), np.array(q_r),
                            right=False)
    # discard the indices greater than number of rings
    mesh[mesh > num_rings] = 0

    # Edge values of each rings
    ring_vals = []

    for i in range(0, num_rings):
        if i < num_rings:
            ring_vals.append(q_r[i])
            ring_vals.append(q_r[i + 1])
        else:
            ring_vals.append(q_r[num_rings-1])

    ring_vals = np.asarray(ring_vals)

    (ring_inds, ring_vals, num_pixels,
     pixel_list) = _process_rings(num_rings, img_dim,
                                  ring_vals, mesh)

    return mesh.reshape(img_dim), ring_inds, ring_vals, num_pixels, pixel_list


def roi_rings_step(img_dim, calibrated_center, num_rings,
               first_r, delta_r, *step_r):
    """
    This will provide the indices of the required rings,
    find the bin edges of the rings, and count the number
    of pixels in each ring, and pixels indices for the
    required rings when there is a step value between rings.
    Step value can be same or different steps between
    each ring.

    Parameters
    ----------
    img_dim: tuple
        shape of the image (detector X and Y direction)
        Order is (num_rows, num_columns)
        shape is [detector_size[0], detector_size[1]])

    calibrated_center : tuple
        defining the center of the image (column value, row value) (mm)

    num_rings : int
        number of rings

    first_r : float
        radius of the first ring

    delta_r : float
        thickness of the ring

    step_r : tuple
        step value for the next ring from the end of the previous
        ring.
        same step - same step values between rings (one value)
        different steps - different step value between rings (provide
        step value for each ring eg: 6 rings provide 5 step values)

    Returns
    -------
    mesh : nedarry
        indices of the required rings
        shape is ([detector_size[0]*detector_size[1]], )

    ring_inds : ndarray
        indices of the required rings

    ring_vals : ndarray
        edge values of the required rings

    num_pixels : ndarray
        number of pixels in each ring

    pixel_list : ndarray
        pixel indices for the required rings

    """
    grid_values = _grid_values(img_dim, calibrated_center)

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

    # indices of rings
    mesh = np.digitize(np.ravel(grid_values), np.array(ring_vals),
                            right=False)

    # to discard every-other bin and set the discarded bins indices to 0
    mesh[mesh % 2 == 0] = 0
    # change the indices of odd number of rings
    indx = mesh > 0
    mesh[indx] = (mesh[indx] + 1) // 2

    (ring_inds, ring_vals, num_pixels,
     pixel_list) = _process_rings(num_rings, img_dim,
                                  ring_vals, mesh)

    return mesh.reshape(img_dim), ring_inds, ring_vals, num_pixels, pixel_list


def _grid_values(img_dim, calibrated_center):
    """
    Parameters
    ----------
    img_dim: tuple
        shape of the image (detector X and Y direction)
        Order is (num_rows, num_columns)
        shape is [detector_size[0], detector_size[1]])

    calibarted_center : tuple
        defining the center of the image (column value, row value) (mm)

    """
    xx, yy = np.mgrid[:img_dim[0], :img_dim[1]]
    x_ = (xx - calibrated_center[1])
    y_ = (yy - calibrated_center[0])
    grid_values = np.float_(np.hypot(x_, y_))

    return grid_values


def _process_rings(num_rings, img_dim, ring_vals, ring_inds):
    """
    This will find the indices of the required rings, find the bin
    edges of the rings, and count the number of pixels in each ring,
    and pixels list for the required rings.

    Parameters
    ----------
    num_rings : int
        number of rings

    img_dim: tuple
        shape of the image (detector X and Y direction)
        Order is (num_rows, num_columns)
        shape is [detector_size[0], detector_size[1]])

    ring_vals : ndarray
        edge values of each ring

    ring_inds : ndarray
        indices of the required rings
        shape is ([detector_size[0]*detector_size[1]], )

    Returns
    -------
    ring_inds : ndarray
        indices of the ring values for the required rings
        (after discarding zero values from the shape
        ([detector_size[0]*detector_size[1]], )

    ring_vals : ndarray
        edge values of each ring
        shape is (num_rings, 2)

    num_pixels : ndarray
        number of pixels in each ring

    pixel_list : ndarray
        pixel indices for the required rings

    """
    # find the pixel list
    w = np.where(ring_inds > 0)
    grid = np.indices((img_dim[0], img_dim[1]))
    pixel_list = (grid[0]*img_dim[1] + grid[1]).flatten()[w]

    ring_inds = ring_inds[ring_inds > 0]

    ring_vals = np.array(ring_vals)
    ring_vals = ring_vals.reshape(num_rings, 2)

    # number of pixels in each ring
    num_pixels = np.bincount(ring_inds, minlength=(num_rings+1))
    num_pixels = num_pixels[1:]

    return ring_inds, ring_vals, num_pixels, pixel_list
