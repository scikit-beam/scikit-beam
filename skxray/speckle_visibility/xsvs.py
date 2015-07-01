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
    This module will provide analysis codes for
    X-ray Speckle Visibility Spectroscopy (XSVS)
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np
from six.moves import zip
from six import string_types

import skxray.correlation as corr
import skxray.roi as roi
import skxray.speckle_analysis as spe_vis
from skxray.core import bin_edges_to_centers

import logging
logger = logging.getLogger(__name__)


def xsvs(image_sets, label_array, timebin_num=2):
    """
    Parameters
    ----------
    sample_dict : array
        sets of images

    label_array : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).

    timebin_num : int, optional
        integration times

    Returns
    -------
    speckle_cts_all : array

    speckle_cts_std_dev : array


    Note
    ----
    These implementation is based on following references
    References: text [1]_, text [2]_

    .. [1] L. Li, P. Kwasniewski, D. Oris, L Wiegart, L. Cristofolini,
       C. Carona and A. Fluerasu , "Photon statistics and speckle visibility
       spectroscopy with partially coherent x-rays" J. Synchrotron Rad.,
       vol 21, p 1288-1295, 2014.

    .. [2] R. Bandyopadhyay, A. S. Gittings, S. S. Suh, P.K. Dixon and
       D.J. Durian "Speckle-visibilty Spectroscopy: A tool to study
       time-varying dynamics" Rev. Sci. Instrum. vol 76, p  093110, 2005.

    """
    max_cts = spe_vis.max_counts(image_sets, label_array)

    # number of ROI's
    num_roi = np.max(label_array)

    # create integration times
    time_bin = spe_vis.time_series(timebin_num, number_of_img)

    # number of items in the time bin
    num_times = len(time_bin)

    labels, indices = corr.extract_label_indices(label_array)
    # number of pixels per ROI
    num_pixels = np.bincount(labels, minlength=(num_roi+1))[1:]
    #num_pixels = num_pixels[1:]

    speckle_cts_all = np.zeros([num_times, num_roi], dtype=np.object)
    speckle_cts_pow_all = np.zeros([num_times, num_roi], dtype=np.object)
    std_dev = np.zeros([num_times, num_roi], dtype=np.object)
    bin_edges = np.zeros((num_times, num_roi), dtype=object)

    for i in range(num_times):
        for j in range(num_roi):
            bin_edges[i, j] =  np.arange(max_cts*2**i )

    for i, images in image_sets:
        # Ring buffer, a buffer with periodic boundary conditions.
        # Images must be keep for up to maximum delay in buf.
        buf = np.zeros([num_times, timebin_num] ,
                       dtype=np.object)  #// matrix of buffers

        # to track processing each level
        track_level = np.zeros(num_times)

        # to increment buffer
        cur = np.ones(num_times)*timebin_num

        # to track how many images processed in each level
        img_per_level = np.zeros(num_times, dtype=np.int64)

        speckle_cts = np.zeros([num_times, num_roi],
                               dtype=np.object)
        speckle_cts_pow = np.zeros([num_times, num_roi],
                                   dtype=np.object)
        for n, img in images:
            cur[0] = (1 + cur[0])%timebin_num
            # read each frame
            # Put the image into the ring buffer.
            buf[0, cur[0] - 1 ] = (np.ravel(img))[indices]

            _process(num_roi, 0, cur[0] - 1, buf, img_per_level, labels, max_cts,
                     bin_edges[0,0], speckle_cts, speckle_cts_pow, i)

            # check whether the number of levels is one, otherwise
            # continue processing the next level
            processing = num_times > 1
            level = 1

            while processing:
                if not track_level[level]:
                    track_level[level] = 1
                    processing = 0
                else:
                    prev =  1 + (cur[level - 1] - 2)%timebin_num
                    cur[level] =  1 + cur[level]%timebin_num

                    buf[level, cur[level]-1] = (buf[level-1,
                                               prev-1] + buf[level-1, cur[level - 1] - 1])
                    track_level[level] = 0

                    _process(num_roi, level, cur[level]-1, buf, img_per_level,
                             labels, max_cts, bin_edges[level, 0], speckle_cts,
                             speckle_cts_pow, i)
                    level += 1
                    # Checking whether there is next level for processing
                    processing = level < num_times


            speckle_cts_all += (speckle_cts -
                                speckle_cts_all)/(i + 1)
            speckle_cts_pow_all += (speckle_cts_pow -
                                    speckle_cts_pow_all)/(i + 1)
            speckle_cts_std_dev = np.power((speckle_cts_all -
                                            np.power(speckle_cts_all, 2)), .5)

    return speckle_cts_all, speckle_cts_std_dev


def _process(num_roi, level, buf_no, buf, img_per_level, labels, max_cts,
             bin_edges, speckle_cts, speckle_cts_pow, i):
    """
    Parameters
    ----------
    num_roi : int

    level : int

    buf_no : int

    buf : array

    img_per_level : int

    labels : array

    max_cts: int

    bin_edges : list

    speckle_cts : array

    speckle_cts_pow : array

    i : int

    """
    img_per_level[level] += 1

    for j in xrange(num_roi):
        roi_data = buf[level, buf_no][labels == j+1 ]

        spe_hist, bin_edges = np.histogram(roi_data, bins=bin_edges,
                                           normed=True)

        speckle_cts[level, j] += (spe_hist -
                                  speckle_cts[level, j] )/(img_per_level[level])

        speckle_cts_pow[level, j] += (np.power(spe_hist, 2) -
                                      speckle_cts_pow[level, j])/(img_per_level[level])

    return None # modifies arguments in place!


def normalize_bin_edges(bin_edges, mean_int_roi):
    """
    Parameters
    ----------
    bin_edges : array
        bin edges for each integration times and each ROI
        shape (number of integration times, number of ROI's)

    mean_int_roi : array
        mean intensity of each ROI
        shape (number of ROI's)

    Returns
    -------
    norm_bin_edges : array
        normalized bin edges
        shape of the bin_edges

    norm_bin_centers :array
        normalized bin centers
        shape of the bin_edges
    """
    num_times, num_rings = bin_edges.shape
    norm_bin_edges = np.zeros((bin_edges.shape), dtype=object)
    norm_bin_centers = np.zeros((bin_edges.shape), dtype=object)
    for i in range(num_times):
        for j in range(num_rings):
            norm_bin_edges[i, j] = bin_edges[i, j]/(mean_int_roi[j]*2**i)
            norm_bin_centers[i, j] = bin_edges_to_centers(norm_bin_edges[i, j])

    return norm_bin_edges, norm_bin_centers
