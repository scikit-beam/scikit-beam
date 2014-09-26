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

This module is for functions specific to 1 time correlation
calculations.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
import logging
logger = logging.getLogger(__name__)
import nsls2.recip as recip
import time

def one_time_corr(num_levels, num_channels, num_qs, img_stack, q_inds):
    """
    Parameters
    ----------
    num_levels: int
        number of levels of multiple-taus

    num_channels: int
        number of channels or number of buffers in auto-correlators
        normalizations (must be even)

    num_qs : int
        number of Q rings

    img_stack : ndarray
        Intensity array of the images
        dimensions are: [num_img][num_rows][num_cols]

    q_inds : ndarray
        indices of the Q values for the required rings

    num_pixels : ndarray
        number of pixels in certain Q ring
        dimensions are : [num_qs]X1

    Returns
    -------
    g2 : ndarray
        matrix of one-time correlation

    Note
    ----
    Standard multiple-tau algorithm is used for one-time intensity
    auto-correlation functions. To evaluate an estimator for the
    correlation function separately for each pixel of the exposed
    detector area needed before analysis. Therefore, standard
    normalization scheme which leads to noise level lower than
    those obtained used for that purpose. (Reference 1)

    References: text [1]_

    .. [1] D. Lumma, L.B. Lurio, S.G.J. Mochrie, and M Sutton,
       "Area detector based photon correlation in the egime of
       short data batches: Data reduction for dynamic x-ray
       scattering," Rev. Sci. Instr., vol 71, pp 3274-3289, 2000.

    """

    if (num_channels% 2 == 0):
        raise ValueError(" Number of channels(buffers) must be even ")

    # total number of channels ( or total number of delay times)
    tot_channels = (num_levels +1 )*num_channels/2

    lag_times =[] # delay ( or lag times)
    lag = np.arange(1, num_channels + 1)
    lag_times.extend(lag)
    for i in range(2, num_levels+1):
        lag = np.array([5, 6, 7, 8])*(2**(i-1))
        lag_times.extend(lag)

    # matrix of auto-correlation function without normalizations
    G = np.zeros((tot_channels, num_qs), dtype = np.float64)
    # matrix of past intensity normalizations
    IAP = np.zeros((tot_channels, num_qs), dtype = np.float64)
    # matrix of future intensity normalizations
    IAF = np.zeros((tot_channels, num_qs), dtype = np.float64)
    # keeps track of number of terms for averaging
    num_terms = np.zeros(num_levels, dtype = np.float64)

    # matrix of one-time correlation
    g2 = np.zeros((tot_channels, num_qs), dtype = np.float64)

    # matrix of buffers
    buf = np.zeros((num_levels, num_channels, no_pixels),
                   dtype = np.float64)
    cur = np.zeros((num_channels, num_levels), dtype = np.float64)
    cts = np.zeros(num_levels)

    num_imgs = img_stack.shape[0] # number of images(frames)
    for i in range(1, num_imgs):

        # delay times for each image
        delay_nums = [x for x in (i - np.array(lag_times)) if x > 0]
        delay_nums.pop(0)

        # buffer numbers
        past_nums = [x for x in (i - np.array(lag_times)) if x > -1]
        past_nums.pop()
        # updating future intensities
        IF = img_stack[delay_nums]
        # updating past intensities
        IP = img_stack[past_nums]
        IFP = IF*IP
        #for j in range (0, len(delay_nums)):
          #  G[j] += np.histogram(np.ravel(IF*IP[j]), q_inds)

    return IFP


def correlation(img_stack, lag_times, q_inds, num_pixels):
    num_imgs = img_stack.shape[0] # number of images(frames)
    for i in range(1, num_imgs):

        # delay times for each image
        delay_nums = [x for x in (i - np.array(lag_times)) if x > 0]
        delay_nums.pop(0)

        # buffer numbers
        past_nums = [x for x in (i - np.array(lag_times)) if x > -1]
        past_nums.pop()
        # updating future intensities
        IF = img_stack[delay_nums]
        # updating past intensities
        IP = img_stack[past_nums]
        IFP = IF*IP
        for j in range (0, len(delay_nums)):
            G[j] += np.histogram(np.ravel(IF*IP[j]), q_inds)
