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
import numpy as np
from . import utils
import logging
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
    return tuple(bins[np.argmax(vals)] for vals, bins in
                 (_corr_ax1(_im) for _im in (input_image.T, input_image)))


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
    norm_mask = np.correlate(m_ones, m_ones, mode='full')
    # not sure that the /2 is the correct correction
    est_by_row = [
        np.argmax(np.correlate(v, v[::-1], mode='full')/norm_mask) / 2
        for v in input_image]
    return np.histogram(est_by_row, bins=np.arange(0, dim + 1))


def construct_circ_avg_image(radii, intensities, dims=None, center=None,
                             pixel_size=(1, 1), left=0, right=0):
    """ Constructs a 2D image from circular averaged data
        where radii are given in units of pixels.
        Normally, data will be taken from circular_average and used to
        re-interpolate into an image.

    Parameters
    ----------
    radii : 1D array of floats
        the radii (must be in pixels)
    intensities : 1D array of floats
        the intensities for the radii
    dims : 2 tuple of floats, optional
        [dy, dx] (row, col)
        dy, dx are the dimensions in row,col format If the dims are not set, it
        will assume the dimensions to be (2*maxr+1) where maxr is the maximum
        radius. Note in the case of rectangular pixels (pixel_size not 1:1)
        that the `maxr' value will be different in each dimension
    center: 2 tuple of floats, optional
        [y0, x0] (row, col)
        y0, x0 is the center (in row,col format)
        if not centered, assumes center is (dims[0]-1)/2., (dims[1]-1)/2.
    pixel_size : tuple, optional
        The size of a pixel (in a real unit, like mm).
        argument order should be (pixel_height, pixel_width)
        default is (1, 1)
    left : float, optional
        pixels smaller than the minimum radius are set to this value
        (set to None for the value at the minimum radius)
    right : float, optional
        pixels larger than the maximum radius are set to this value
        (set to None for the value at the maximum radius)

    Returns
    -------
    IMG : the interpolated circular averaged image

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
            raise ValueError("Specifying a dims but not a center does not "
                             "make sense and may lead to unexpected results.")

        # round up, also take into account pixel size change
        maxr_y, maxr_x = (int(np.max(radii/pixel_size[0])+.5),
                          int(np.max(radii/pixel_size[1])+.5))
        dims = 2*maxr_y+1, 2*maxr_x+1

    if center is None:
        center = (dims[0]-1)/2., (dims[1] - 1)/2.

    radial_val = utils.radial_grid(center, dims, pixel_size)
    CIMG = np.zeros(dims)
    CIMG = np.interp(radial_val, radii, intensities, right=0)

    return CIMG
