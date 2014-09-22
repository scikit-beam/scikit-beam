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

def one_time_corr(num_levels, num_channels, num_qs, img_stack, pixel_list, q_inds):
    """
    Parameters
    ----------
    num_levels: int
        number of levels of multiple-taus

    num_channels: int
        number of channels or number of buffers in auto-correlators
        normalizations (must be even)

    num_qs : int

    img_stack : ndarray

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

    .. [1] D. Lumma,L.B. Lurio, S.G.J. Mochrie, and M Sutton, "Area detector
       based photon correlation in the egime of short data batches:
       Data reduction for dynamic x-ray scattering," Rev. Sci. Instr.,
       vol 71, pp 3274-3289, 2000.

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

    # matrix of buffers
    buf = np.zeros((num_levels, num_channels, no_pixels), dtype = np.float64)
    cur = np.zeros((num_channels, num_levels), dtype = np.float64)
    cts = np.zeros(num_levels)

    num_imgs = img_stack.shape[0]
    for i in range(0, num_imgs):
        cur[1] = 1 + cur[1]%num_channels
        insert_img(i, img_stack, num_channels, pixel_list, q_inds, cur)
        img = img_stack[i]
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
        delay_num = 1 + (buf_num - (i -1)-1 + num_channels)%buf_num

        IP = buf[lev, delay_num, ]
        IF = buf[lev, buf_num, ]
        G[ptr, ] +=
        IAP[ptr, ] +=
        IAF[ptr, ] +=


def q_values(detector_size, pixel_size, dist_sample,
             calibrated_center,wavelength, num_qs,
             first_q, step_q, delta_q):
    """
    This will compute the Q space values for all pixels in a
    shape specified by detector_size. The

    Parameters
    ----------
    detector_size : tuple
        2 element tuple defining no. of pixels(size) in the
        detector X and Y direction(mm)

    pixel_size : tuple
       2 element tuple defining the (x y) dimensions of the
       pixel (mm)

    dist_sample : float
       distance from the sample to the detector (mm)

    calibrated_center : tuple
        2 element tuple defining the (x y) center of the
        detector (mm)

    wavelength : float
        wavelength of incident radiation (Angstroms)

    num_qs : int
        number pf q rings

    first_q : float
        q value of the first q ring

    step_q : float
        step value for next q ring

    delta_q : float
        thickness of the q ring

    Returns
    -------
    q_values : ndarray
        hkl values

    q_inds : ndarray
        indices of the q values for the required rings

    num_pixels : ndarray
        number of pixels in certain q ring
        1*[num_qs]


    """
    # setting angles of the detector
    # delta=40, theta=15, chi = 90, phi = 30, mu = 10.0, gamma=5.0
    setting_angles = np.array([0., 0., 0., 0., 0., 0.])

    # UB matrix (orientation matrix) 3x3 matrix
    ub_mat = np.identity(3)

    hkl_val = recip.process_to_q(setting_angles, detector_size,
                                 pixel_size, calibrated_center,
                                 dist_sample, wavelength, ub_mat)
    q_val = np.sqrt(hkl_val[:,0]**2 + hkl_val[:,1]**2 + hkl_val[:,2]**2)
    q_values = q_val.reshape(detector_size[0], detector_size[1])

    q_ring_val = []
    q = first_q
    q_ring_val.append(first_q)
    for i in range(1,num_qs):
        q += (step_q + delta_q)
        q_ring_val.append(q)

    q_ring_val = np.array(q_ring_val)
    q_values = np.ravel(q_values)
    q_inds = np.digitize(np.ravel(q_values), np.array(q_ring_val))

    num_pixels = np.bincount(q_inds)

    return q_values, q_inds, num_pixels
