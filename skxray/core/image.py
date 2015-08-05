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
import six
import logging
logger = logging.getLogger(__name__)
import numpy as np


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
    est_by_row = [np.argmax(np.correlate(v, v[::-1],
                                         mode='full')/norm_mask) / 2
             for v in input_image]
    return np.histogram(est_by_row, bins=np.arange(0, dim + 1))

def hist_make(src_data,
              num_bins,
              pd_function=False):
    """
    This function evaluates the histogram of the source data set and returns
    bin data consisting of both bin edges and bin averages. This tool is
    primarily geared for plotting histograms for visual analysis and
    comparison.

    Parameters
    ----------
    src_data : ndarray
        Can be JxK or IxJxK
        Specifies the source data set from which you want to evaluate the
        histogram.
    num_bins : int
        Specify the number of bins to include in the histogram as an integer.

    pd_function : bool, optional
        Identify whether the histogram data should be normalized as a
        probability density histogram or not.
        Options:
            True -- Histogram data is normalized to range from 0 to 1
            False -- Histogram data reported simply as "counts" (e.g. Voxel
                     Count)

    Returns
    -------
    hist : array
        1xN array containing all of the actual bin measurements (e.g. voxel
        counts)
    bin_avg : array
        1xN array containing the average intensity value for each bin.
        NOTE: the length of this array is equal to the length of the hist
        array
    bin_edges : array
        1xN array containing the edge values for each bin
        NOTE: the length of this array is 1 larger than the length of the
        hist array (e.g. len(bin_edges) = len(hist) + 1)
    """

    hist, bin_edges = np.histogram(src_data,
                                   bins=num_bins,
                                   density=pd_function)
    bin_avg = np.empty(len(hist))
    intensity = iter(bin_edges)
    row_count = 0
    for left_bin_edge in bin_edges:
        right_bin_edge = next(intensity)
        bin_avg[row_count] = (left_bin_edge + right_bin_edge) / 2
        row_count += 1
        if right_bin_edge == bin_edges[len(bin_edges) - 1]:
            break
    return hist, bin_edges, bin_avg