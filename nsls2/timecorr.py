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
from core import bin_1D
import time


def one_time_corr(num_levels, num_channels, num_qs, img_stack, pixel_list, q_inds):
    """
    Parameters
    ----------
    num_levels: int
        number of levels of multiple-taus

    num_channels: int
        number of channels or number of buffers in auto-correlators
        normalizations (must be even)

    Returns
    -------

    Note
    ----
    To evaluate an estimator for the correlation function separately for
    each pixel of the exposed detector area needed before analysis.
    Therefore, standard normalization scheme which leads to noise level
    lower than those obtained used for that purpose. (Reference 1)

    Multi-tau

    References: text [1]_

    .. [1] D. Lumma,L.B. Lurio, S.G.J. Mochrie, and M Sutton, "Area detector
       based photon correlation in the egime of short data batches:
       Data reduction for dynamic x-ray scattering," Rev. Sci. Instr.,
       vol 71, pp 3274-3289, 2000.

    ..[2]


    """

    if (num_channels% 2 == 0):
        raise ValueError(" Number of channels(buffers) must be even ")

    # total number of channels ( or total number of delay times)
    tot_channels = (num_levels +1 )*num_channels
    # num_channels + 4*(num_levels-1)
    lag_times =[] # delay ( or lag times)
    lag = np.arange(num_channels)
    lag_times.append(lag)
    for i in range(2, num_levels+1):
        lag = np.array([4, 5, 6, 7])*(2**(i-1))
        lag_times.append(lag)

    #
    # G=array(0.0,(nolev+1)*nobuf/2,noqs)
	# IAP=array(0.0,(nolev+1)*nobuf/2,noqs)
	# IAF=array(0.0,(nolev+1)*nobuf/2,noqs)
	# num=array(0,nolev)
    #buf=array(0.0,nolev,nobuf,nopixels) // matrix of buffers
	#cts=array(0,nolev)
	#cur=array(nobuf,nolev)


    # matrix of cross-correlations
    G = np.zeros((tot_channels, num_qs), dtype = np.float64)
    # matrix of past intensity normalizations
    IAP = np.zeros((tot_channels, num_qs), dtype = np.float64)
    # matrix of future intensity normalizations
    IAF = np.zeros((tot_channels, num_qs), dtype = np.float64)
    # keeps track of number of terms for averaging
    num_terms = np.zeros(num_levels, dtype = np.float64)

    # matrix of one-time correlation
    g2 = np.zeros((tot_channels, num_qs), dtype = np.float64)
    #

    # matrix of buffers
    buf = np.zeros((num_levels, num_channels, no_pixels), dtype = np.float64)
    cur = np.zeros((num_channels, num_levels), dtype = np.float64)
    cts = np.zeros(num_levels)

    num_imgs = img_stack.shape[0]
    #cur = 0
    for i in range(0, num_imgs):
        cur[1] = 1 + cur[1]%num_channels
        insert_img(i, img_stack, num_channels, pixel_list, q_inds, cur)
        img = img_stack[n]
        imin = (1 + num_channels/2)
        process(1, cur[1], num_terms, imin, num_channels)
        processing =1
        lev = 2
        while(processing == 1):
            if cts[lev]:
                prev = 1 + (cur(lev - 1) -1 -1 + num_channels)%num_channels
                cur[lev] = 1 + cur[lev]%num_channels
                buf()
                cts[lev] = 0
                process(lev, cur[lev], num_terms, imin, num_channels)
                lev += 1
                if (lev > num_levels):
                    processing = 0
            else:
                cts[lev] = 1
                processing = 0


def process(lev, buf_num, num_terms, imin, num_channels):
    num_terms[lev] +=1

    for i in range(imin, min(num_terms(lev), num_channels): # loop over delays
        ptr = (lev - 1)* num_channels/2 + i
        delay_num = 1 + (buf_num - (i -1)-1 + buf_num)%buf_num

        IP = buf[lev, delay_num, ]
        IF = buf[lev, buf_num, ]
        G[ptr, ] +=
        IAP[ptr, ] +=
        IAF[ptr, ] +=


def insert_img(n, img_stack, num_channels, pixel_list, q_inds, cur):
    img = img_stack[n]
    process(1,cur)
    processing = 1
    lev = 2
    pass



def pixelist():
    pass
