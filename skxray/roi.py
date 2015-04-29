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
This module contain convenience methods to generate ROI labeled arrays for
simple shapes such as rectangles and concentric circles.
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six

from six.moves import zip
from six import string_types

import time
import collections
import sys
import numpy as np
import numpy.ma as ma

import skxray.correlation as corr
import skxray.core as core

import logging
logger = logging.getLogger(__name__)


def rectangles(coords, shape):
    """
    This function wil provide the indices array for rectangle region of interests.

    Parameters
    ----------
    coords : iterable
        coordinates of the upper-left corner and width and height of each
        rectangle: e.g., [(x, y, w, h), (x, y, w, h)]

    shape : tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. Order is (rr, cc).

    Returns
    -------
    label_array : array
        Elements not inside any ROI are zero; elements inside each
        ROI are 1, 2, 3, corresponding to the order they are specified
        in coords. Order is (rr, cc).

    """

    labels_grid = np.zeros(shape, dtype=np.int64)

    for i, (col_coor, row_coor, col_val, row_val) in enumerate(coords):

        left, right = np.max([col_coor, 0]), np.min([col_coor + col_val,
                                                     shape[0]])
        top, bottom = np.max([row_coor, 0]), np.min([row_coor + row_val,
                                                     shape[1]])

        area = (right - left) * (bottom - top)

        slc1 = slice(left, right)
        slc2 = slice(top, bottom)

        if np.any(labels_grid[slc1, slc2]):
            raise ValueError("overlapping ROIs")

        # assign a different scalar for each roi
        labels_grid[slc1, slc2] = (i + 1)

    return labels_grid


def rings(edges, center, shape):
    """
    Draw annual (ring-shaped) regions of interest.

    Each ring will be labeled with an integer. Regions outside any ring will
    be filled with zeros.

    Parameters
    ----------
    edges: list
        giving the inner and outer radius of each ring
        e.g., [(1, 2), (11, 12), (21, 22)]

    center : tuple
        point in image where r=0; may be a float giving subpixel precision.
        Order is (rr, cc).

    shape: tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. Order is (rr, cc).

    Returns
    -------
    label_array : array
        Elements not inside any ROI are zero; elements inside each
        ROI are 1, 2, 3, corresponding to the order they are specified
        in edges.
    """
    edges = np.atleast_2d(np.asarray(edges)).ravel()
    if not 0 == len(edges) % 2:
        raise ValueError("edges should have an even number of elements, "
                         "giving inner, outer radii for each ring")
    if not np.all(np.diff(edges) >= 0):
        raise ValueError("edges are expected to be monotonically increasing, "
                         "giving inner and outer radii of each ring from "
                         "r=0 outward")
    r_coord = core.radial_grid(center, shape).ravel()
    label_array = np.digitize(r_coord, edges, right=False)
    # Even elements of label_array are in the space between rings.
    label_array = (np.where(label_array % 2 != 0, label_array, 0) + 1) // 2
    return label_array.reshape(shape)


def ring_edges(inner_radius, width, spacing=0, num_rings=None):
    """
    Calculate the inner and outer radius of a set of rings.

    The number of rings, their widths, and any spacing between rings can be
    specified. They can be uniform or varied.

    Parameters
    ----------
    inner_radius : float
        inner radius of the inner-most ring

    width : float or list of floats
        ring thickness
        If a float, all rings will have the same thickness.

    spacing : float or list of floats, optional
        margin between rings, 0 by default
        If a float, all rings will have the same spacing. If a list,
        the length of the list must be one less than the number of
        rings.

    num_rings : int, optional
        number of rings
        Required if width and spacing are not lists and number
        cannot thereby be inferred. If it is given and can also be
        inferred, input is checked for consistency.

    Returns
    -------
    edges : array
        inner and outer radius for each ring

    Example
    -------
    # Make two rings starting at r=1px, each 5px wide
    >>> ring_edges(inner_radius=1, width=5, num_rings=2)
    [(1, 6), (6, 11)]
    # Make three rings of different widths and spacings.
    # Since the width and spacings are given individually, the number of 
    # rings here is simply inferred.
    >>> ring_edges(inner_radius=1, width=(5, 4, 3), spacing=(1, 2))
    [(1, 6), (7, 11), (13, 16)]
    """
    # All of this input validation merely checks that width, spacing, and
    # num_rings are self-consistent and complete.
    width_is_list = isinstance(width, collections.Iterable)
    spacing_is_list = isinstance(spacing, collections.Iterable)
    if (width_is_list and spacing_is_list):
        if len(width) != len(spacing) - 1:
            raise ValueError("List of spacings must be one less than list "
                             "of widths.")
    if num_rings is None:
        try:
            num_rings = len(width)
        except TypeError:
            try:
                num_rings = len(spacing) + 1
            except TypeError:
                raise ValueError("Since width and spacing are constant, "
                                 "num_rings cannot be inferred and must be "
                                 "specified.")
    else:
        if width_is_list:
            if num_rings != len(width):
                raise ValueError("num_rings does not match width list")
        if spacing_is_list:
            if num_rings-1 != len(spacing):
                raise ValueError("num_rings does not match spacing list")

    # Now regularlize the input.
    if not width_is_list:
        width = np.ones(num_rings) * width
    if not spacing_is_list:
        spacing = np.ones(num_rings - 1) * spacing

    # The inner radius is the first "spacing."
    all_spacings = np.insert(spacing, 0, inner_radius)
    steps = np.array([all_spacings, width]).T.ravel()
    edges = np.cumsum(steps).reshape(-1, 2)

    return edges


def segmented_rings(edges, segments, center, shape, offset_angle=0):
    """
    Parameters
    ----------
    edges : array
         inner and outer radius for each ring

    segments : int or list
        number of pie slices or list of angles in radians
        That is, 8 produces eight equal-sized angular segments,
        whereas a list can be used to produce segments of unequal size.

    center : tuple
        point in image where r=0; may be a float giving subpixel precision.
        Order is (rr, cc).

    shape: tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. Order is (rr, cc).

    angle_offset : float or array, optional
        offset in radians from offset_angle=0 along the positive X axis

    Returns
    -------
    label_array : array
        Elements not inside any ROI are zero; elements inside each
        ROI are 1, 2, 3, corresponding to the order they are specified
        in edges and segments

    """
    edges = np.asarray(edges).ravel()
    if not 0 == len(edges) % 2:
        raise ValueError("edges should have an even number of elements, "
                         "giving inner, outer radii for each ring")
    if not np.all(np.diff(edges) >= 0):
        raise ValueError("edges are expected to be monotonically increasing, "
                         "giving inner and outer radii of each ring from "
                         "r=0 outward")

    agrid = core.angle_grid(center, shape)

    agrid[agrid < 0] = 2*np.pi + agrid[agrid < 0]

    segments_is_list = isinstance(segments, collections.Iterable)
    if segments_is_list:
        segments = np.asarray(segments) + offset_angle
    else:
        # N equal segments requires N+1 bin edges spanning 0 to 2pi.
        segments = np.linspace(0, 2*np.pi, num=1+segments, endpoint=True)
        segments += offset_angle

    # the indices of the bins(angles) to which each value in input
    #  array(angle_grid) belongs.
    ind_grid = (np.digitize(np.ravel(agrid), segments,
                            right=False)).reshape(shape)

    label_array = np.zeros(shape, dtype=np.int64)
    # radius grid for the image_shape
    rgrid = core.radial_grid(center, shape)

    # assign indices value according to angles then rings
    len_segments = len(segments)
    for i in range(len(edges) // 2):
        indices = (edges[2*i] <= rgrid) & (rgrid < edges[2*i + 1])
        # Combine "segment #" and "ring #" to get unique label for each.
        label_array[indices] = ind_grid[indices] + (len_segments - 1) * i

    return label_array
