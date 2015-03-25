# ######################################################################
# Original code(in Yorick):                                            #
# @author: Mark Sutton                                                 #
#                                                                      #
# Developed at the NSLS-II, Brookhaven National Laboratory             #
# Developed by Sameera K. Abeykoon, February 2014                      #
#                                                                      #
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

This module is for functions specific to time correlation

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np
import logging
logger = logging.getLogger(__name__)
import time

import skxray.core as core


def auto_corr(num_levels, num_bufs, num_qs,
              pixel_list, q_inds, img_stack):
    """
    This module is for one time correlation.
    The multi-tau correlation scheme was used for
    finding the lag times (delay times).

    Parameters
    ----------
    num_levels : int
        number of levels of multiple-taus

    num_bufs : int, even
        number of channels or number of buffers in
        multiple-taus (must be even)

    num_qs : int
        number of region of interests(roi's)

    pixel_list : ndarray
        pixel indices for the required roi's

    q_inds : ndarray
        indices of the required roi's

    img_stack : ndarray
        intensity array of the images
        dimensions are: (num of images, num_rows, num_cols)

    Returns
    -------
    g2 : ndarray
        matrix of one-time correlation
        shape (num_levels, num_qs)

    lag_steps : ndarray
        delay or lag steps for the multiple tau analysis
        shape num_levels

    Notes
    -----
    In order to calculate correlations for delays, images must be
    kept for up to the maximum delay. These are stored in the array
    buf. This algorithm only keeps number of buffers and delays but
    several levels of delays number of levels are kept in buf. Each
    level has twice the delay times of the next lower one. To save
    needless copying, of cyclic storage of images in buf is used.

    References: text [1]_

    .. [1] D. Lumma, L. B. Lurio, S. G. J. Mochrie and M. Sutton,
        "Area detector based photon correlation in the regime of
        short data batches: Data reduction for dynamic x-ray
        scattering," Rev. Sci. Instrum., vol 70, p 3274-3289, 2000.

    """
    if num_bufs%2 != 0:
        raise ValueError("number of channels or number of buffers in"
                         " multiple-taus (must be even)")

    # number of pixels in required roi's, dimensions are : [num_qs]X1
    num_pixels = np.bincount(q_inds, minlength=(num_qs+1))
    num_pixels = num_pixels[1:]

    if np.any(num_pixels == 0):
        raise ValueError("Number of pixels of the required roi's"
                         " cannot be zero, "
                         "num_pixels = {0}".format(num_pixels))

    # matrix of auto-correlation function without normalizations
    G = np.zeros(((num_levels + 1)*num_bufs/2, num_qs),
                 dtype=np.float64)

    # matrix of past intensity normalizations
    IAP = np.zeros(((num_levels + 1)*num_bufs/2, num_qs),
                   dtype=np.float64)

    # matrix of future intensity normalizations
    IAF = np.zeros(((num_levels + 1)*num_bufs/2, num_qs),
                   dtype=np.float64)

    # matrix of one-time correlation for required roi's
    g2 = np.zeros((num_levels, num_qs), dtype=np.float64)

    # correlation for delays, images must be keep for up to maximum
    # delay in buf
    buf = np.zeros((num_levels, num_bufs, np.sum(num_pixels)),
                   dtype=np.float64)

    # to track processing each level
    cts = np.zeros(num_levels)

    # to increment buffer
    cur = np.ones((num_levels), dtype=np.int64)

    # to track how many images processed in each level
    num = np.array(np.zeros(num_levels), dtype=np.int64)

    # starting time for the process
    t1 = time.time()

    for n in range(0, len(img_stack)):  # changed the number of frames

        cur[0] = (1 + cur[0]) % num_bufs  # increment buffer

        # add image data to the buf to use for correlation
        buf[0, cur[0] - 1] = (np.ravel(img_stack[n]))[pixel_list]

        # call the _process function for multi-tau level one
        G, IAP, IAF, num = _process(buf, G, IAP, IAF, q_inds,
                                    num_bufs, num_pixels, num, level=0,
                                    buf_no=cur[0] - 1)

        # check whether the number of levels is one, otherwise
        # continue processing the next level
        if num_levels > 1:
            processing = 1
        else:
            processing = 0

        # the image data will be saved in buf according to each level then call
        #  _process function to calculate one time correlation functions
        level = 1
        while processing:
            if cts[level]:
                prev = 1 + (cur[level - 1] - 1 - 1 + num_bufs)%num_bufs
                cur[level] = 1 + cur[level]%num_bufs
                buf[level, cur[level] - 1] = (buf[level - 1, prev - 1] +
                                              buf[level - 1,
                                                  cur[level - 1] - 1])/2

                # make the cts zero once that level is processed
                cts[level] = 0

                # call the _process function for each multi-tau level
                # for multi-tau levels greater than one
                G, IAP, IAF, num = _process(buf, G, IAP, IAF, q_inds,
                                       num_bufs, num_pixels, num,
                                       level=level, buf_no=cur[level]-1,)
                level += 1

                # Checking whether there is next level for processing
                if level<num_levels:
                    processing = 1
                else:
                    processing = 0
            else:
                cts[level] = 1
                processing = 0

    # ending time for the process
    t2 = time.time()

    logger.info("Processing time for {0} images took {1} seconds."
                "".format(len(img_stack), (t2-t1)))

    if len(np.where(IAP == 0)[0]) != 0:
        g_max = np.where(IAP == 0)[0][0]
    else:
        g_max = IAP.shape[0]

    # calculate the one time correlation
    g2 = (G[: g_max] / (IAP[: g_max] * IAF[: g_max]))

    # finding the lag times (delay times) for multi-tau levels
    tot_channels, lag_steps = core.multi_tau_lags(num_levels,
                                                  num_bufs)
    lag_steps = lag_steps[:g2.shape[0]]

    return g2, lag_steps


def _process(buf, G, IAP, IAF, q_inds, num_bufs,
             num_pixels, num, level, buf_no):
    """
    This helper function calculates G, IAP and IAF at
    each level, symmetric normalization is used.

    Parameters
    ----------
    buf : ndarray
        image data array to use for correlation

    G : ndarray
        matrix of auto-correlation function without
        normalizations

    IAP : ndarray
        matrix of past intensity normalizations

    IAF : ndarray
        matrix of future intensity normalizations

    q_inds : ndarray
        indices of the required roi's

    num_bufs : int, even
        number of buffers(channels)

    num_pixels : ndarray
        number of pixels in certain roi's
        roi's, dimensions are : [num_qs]X1

    num : ndarray
        to track how many images processed in each level

    level : int
        the current level number

    buf_no : int
        the current buffer number

    Returns
    -------
    G : ndarray
        matrix of auto-correlation function without normalizations

    IAP : ndarray
        matrix of past intensity normalizations

    IAF : ndarray
        matrix of future intensity normalizations

    Notes
    -----
    :math ::
        G   = <I(t)I(t + delay)>

    :math ::
        IAP = <I(t)>

    :math ::
        IAF = <I(t + delay)>

    """
    num[level] += 1

    if level == 0:
        i_min = 0
    else:
        i_min = int(num_bufs/2)

    for i in range(i_min, min(num[level], num_bufs)):
        t_index = level*num_bufs/2 + i

        delay_no = (buf_no - i) % num_bufs

        IP = buf[level, delay_no]
        IF = buf[level, buf_no]

        G[t_index] += ((np.bincount(q_inds,
                                    weights=np.ravel(IP*IF))[1:])/num_pixels
                       - G[t_index])/(num[level] - i)
        IAP[t_index] += ((np.bincount(q_inds,
                                      weights=np.ravel(IP))[1:])/num_pixels
                         - IAP[t_index])/(num[level] - i)
        IAF[t_index] += ((np.bincount(q_inds,
                                      weights=np.ravel(IF))[1:])/num_pixels
                         - IAF[t_index])/(num[level] - i)

    return G, IAP, IAF, num
