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
This is the module for putting advanced/x-ray specific image
processing tools.  These should be interesting compositions of existing
tools, not just straight wrapping of np/scipy/scikit images.
"""
from __future__ import absolute_import, division, print_function

import logging

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import utils

logger = logging.getLogger(__name__)


def find_ring_center_acorr_1D(input_image):
    """
    Find the pixel-resolution center of a set of concentric rings.

    This function uses correlation between the image and it's mirror
    to find the approximate center of  a single set of concentric rings.
    It is assumed that there is only one set of rings in the image.  For
    this method to work well the image must have significant mirror-symmetry
    in both dimensions.

    Parameters
    ----------
    input_image : ndarray
        A single image.

    Returns
    -------
    calibrated_center : tuple
        Returns the index (row, col) of the pixel that rings
        are centered on.  Accurate to pixel resolution.
    """
    return tuple(bins[np.argmax(vals)] for vals, bins in (_corr_ax1(_im) for _im in (input_image.T, input_image)))


def _corr_ax1(input_image):
    """
    Internal helper function that finds the best estimate for the
    location of the vertical mirror plane.  For each row the maximum
    of the correlating with it's mirror is found.  The most common value
    is reported back as the location of the mirror plane.

    Parameters
    ----------
    input_image : ndarray
        The input image

    Returns
    -------
    vals : ndarray
        histogram of what pixel has the highest correlation

    bins : ndarray
        Bin edges for the vals histogram
    """
    dim = input_image.shape[1]
    m_ones = np.ones(dim)
    norm_mask = np.correlate(m_ones, m_ones, mode="full")
    # not sure that the /2 is the correct correction
    est_by_row = [np.argmax(np.correlate(v, v[::-1], mode="full") / norm_mask) / 2 for v in input_image]
    return np.histogram(est_by_row, bins=np.arange(0, dim + 1))


def construct_circ_avg_image(radii, intensities, dims=None, center=None, pixel_size=(1, 1), left=0, right=0):
    """
    Constructs a 2D image from circular averaged data
    where radii are given in units of pixels.
    Normally, data will be taken from circular_average and used to
    re-interpolate into an image.

    Parameters
    ----------
    radii: 1D array of floats
        the radii (must be in pixels)
    intensities: 1D array of floats
        the intensities for the radii
    dims: 2 tuple of floats, optional
        ``[dy, dx] (row, col)``:
        dy, dx are the dimensions in row,col format If the dims are not set, it
        will assume the dimensions to be (2*maxr+1) where maxr is the maximum
        radius. Note in the case of rectangular pixels (pixel_size not 1:1)
        that the ``maxr`` value will be different in each dimension
    center: 2 tuple of floats, optional
        [y0, x0] (row, col)
        y0, x0 is the center (in row,col format)
        if not centered, assumes center is ``(dims[0]-1)/2., (dims[1]-1)/2``.
    pixel_size: tuple, optional
        The size of a pixel (in a real unit, like mm).
        argument order should be (pixel_height, pixel_width)
        default is ``(1, 1)``
    left: float, optional
        pixels smaller than the minimum radius are set to this value
        (set to None for the value at the minimum radius)
    right: float, optional
        pixels larger than the maximum radius are set to this value
        (set to None for the value at the maximum radius)

    Returns
    -------
    IMG
        the interpolated circular averaged image

    See Also
    --------
    circular_average : compute circular average of an image
        Pixels smaller than the minimum radius are set to the value at that
        minimum radius.
        Pixels larger than the maximum radius are set to zero.
    bin_grid : Bin and integrate an image, given the radial array of pixels
        Useful for nonlinear spacing (Ewald curvature)

    Notes
    -----
    Some pixels may not be filled if the dimensions chosen are too large.
    Run this code again on a list of values equal to 1 to obtain a mask
    (and set left=0 and right=0).
    """
    if dims is None:
        if center is not None:
            raise ValueError(
                "Specifying a dims but not a center does not " "make sense and may lead to unexpected results."
            )

        # round up, also take into account pixel size change
        maxr_y, maxr_x = (int(np.max(radii / pixel_size[0]) + 0.5), int(np.max(radii / pixel_size[1]) + 0.5))
        dims = 2 * maxr_y + 1, 2 * maxr_x + 1

    if center is None:
        center = (dims[0] - 1) / 2.0, (dims[1] - 1) / 2.0

    radial_val = utils.radial_grid(center, dims, pixel_size)
    CIMG = np.interp(radial_val, radii, intensities, right=0)

    return CIMG


def construct_rphi_avg_image(radii, angles, image, mask=None, center=None, shape=None, pixel_size=(1, 1)):
    """
    Construct a 2D Cartesian (x,y) image from a polar coordinate image.

    Assumes a 2D array of data. If data is missing, use mask.

    Parameters
    ----------
    radii: 1d array of coordinates, ascending order.
        The radii values (in units of pixels)
        These may be non-uniformly spaced.

    angles: 1d array of coordinates, ascending order
        The angle values (in units of radians)
        Domain *must* be monotonically increasing (or a subset) if
        we plot the reconstructed image in a typical x-y plane,
        where x are the columns of the image and y are the rows
        (img[y,x]), then 0 degrees is the -x direction, and
        ascending angle goes clockwise in this plane.
        These may be non-uniformly spaced.
        Note that if the domain interval exceeds 2*pi, values
        outside this range are ignored for interpolation.
        Example: ``angles = [0, ..., 6.25, 2*np.pi]``
        The last point modulo 2*pi is 0 which is the same as first
        point so it is ignored

    image: 2d array the image to interpolate from
        rows are radii, columns are angles, ex: img[radii, angles]

    mask: 2d array, optional
        the masked data (0 masked, 1 not masked). Defaults to None,
        which means assume all points are valid.

    center: 2 tuple of floats, optional

    shape:
        the new image shape, in terms of pixel values

    Notes
    -----
    This function uses a simple linear interpolation from scipy:
    `scipy.interpolate.RegularGridInterpolator`. More complex interpolation techniques
    (i.e. splines) cannot be used with this algorithm.

    Returns
    -------
    new_img: 2d np.ndarray
        The reconstructed image. Masked regions are filled with `np.nan`.
    """
    if mask is not None:
        if mask.shape != image.shape:
            if mask.ndim == 2:
                raise ValueError(
                    "Mask shape ({}, {}) ".format(*mask.shape)
                    + "does not match expected"
                    + " shape of ({},{})".format(*shape)
                )
            else:
                raise ValueError("Mask not right dimensions." "Expected 2 got {}".format(mask.ndim))
        image[np.where(mask == 0)] = np.nan
    if shape is None:
        if center is not None:
            raise ValueError(
                "Specifying a shape but not a center does not " "make sense and may lead to unexpected results."
            )
        # round up, also take into account pixel size change
        maxr_y, maxr_x = (int(np.max(radii / pixel_size[0]) + 0.5), int(np.max(radii / pixel_size[1]) + 0.5))
        shape = 2 * maxr_y + 1, 2 * maxr_x + 1

    if center is None:
        center = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0

    # 1 . There are a few cases to consider here for the angles:
    #   i. [0,..., 2*pi] -> [0,..,0]
    #   ii. [0, ... , 2*pi, ...5*pi] -> [0, ..., 0, ...]
    #   iii. [0,..., a < 2*pi] -> [0, ..., a< 2*pi] (easy)
    #   iv. [.12, ..., 2*pi, ...] -> [0, ...]

    # 1.a : subtract minimum
    anglemin = angles[0]
    angles -= anglemin
    # 1.b : modulo 2*pi
    angles = angles % (2 * np.pi)
    # 1.c : find any extra cross-overs and ignore them
    adiff = np.where(np.diff(angles) < 0)[0]
    if len(adiff) > 0:
        errorstr = "Error, domain exceeds 2*pi\n"
        errorstr += "Hint : common error is to "
        errorstr += "use np.linspace(0, 2*np.pi, 100), for example\n"
        errorstr += "Use np.linspace(0, 2*np.pi, 100, endpoint=False)"
        errorstr += " instead\n"

        raise ValueError(errorstr)

    # 2 : since the interpolation will be linear, and the angles wrap, we
    # need to add the first angle position to the end and vice versa
    # 2.a : add bounds to angle
    anglesp = np.concatenate(([angles[-1] - 2 * np.pi], angles, [angles[0] + 2 * np.pi]))
    # 2.b : add bounds to image
    imagep = np.zeros((image.shape[0], image.shape[1] + 2))
    imagep[:, 0] = image[:, -1]
    imagep[:, 1 : image.shape[1] + 1] = image
    imagep[:, -1] = image[:, 0]

    radial_val = utils.radial_grid(center, shape, pixel_size).ravel()
    angle_val = utils.angle_grid(center, shape, pixel_size).ravel()
    # 1.d : subtract minimum for interpolated values as well
    angle_val -= anglemin
    angle_val = angle_val % (2 * np.pi)

    interpolator = RegularGridInterpolator((radii, anglesp), imagep, bounds_error=False, fill_value=np.nan)

    new_img = interpolator((radial_val, angle_val))
    new_img = new_img.reshape(shape)

    return new_img
