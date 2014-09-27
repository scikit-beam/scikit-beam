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

    This is an example of using real experimental XPCS(X-ray Photon
    Correlation Spectroscopy) data for calculate one time correlation
    functions.

"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
import logging
logger = logging.getLogger(__name__)
import time
import nsls2.recip as recip
# import nsls2.timecorr as timecorr
import matplotlib.pyplot as plt


def plot_q_rings(q_inds, detector_size):
    q_inds = q_inds.reshape(detector_size[0], detector_size[1])
    plt.title("  Required Q Rings  ")
    plt.imshow(q_inds)
    plt.show()


def one_time_corr(num_levels, num_channels, num_qs, img_stack,
                  q_inds, num_pixels, lag_time):
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

    if (num_channels % 2 != 0):
        raise ValueError(" Number of channels(buffers) must be even ")

    # total number of channels ( or total number of delay times)
    tot_channels = (num_levels + 1)*num_channels/2

    lag_steps = []  # delay ( or lag times)
    lag = np.arange(1, num_channels + 1)
    lag_steps.extend(lag)
    for i in range(2, num_levels+1):
        lag = np.array([5, 6, 7, 8])*(2**(i-1))
        lag_steps.extend(lag)

    # lag_steps = sorted(lag_steps, reverse=True)

    # matrix of auto-correlation function without normalizations
    G = np.zeros((tot_channels, num_qs), dtype=np.float64)
    # matrix of past intensity normalizations
    IAP = np.zeros((tot_channels, num_qs), dtype=np.float64)
    # matrix of future intensity normalizations
    IAF = np.zeros((tot_channels, num_qs), dtype=np.float64)

    # matrices of G, IAF and IAP with all pixels
    GD = np.zeros((tot_channels, num_qs+2), dtype=np.float64)
    IAPD = np.zeros((tot_channels, num_qs+2), dtype=np.float64)
    IAFD = np.zeros((tot_channels, num_qs+2), dtype=np.float64)

    # matrices of G, IAF and IAP after deleting unwanted pixels
    G_del = np.zeros((tot_channels, num_qs), dtype=np.float64)
    IAP_del = np.zeros((tot_channels, num_qs), dtype=np.float64)
    IAF_del = np.zeros((tot_channels, num_qs), dtype=np.float64)

    # matrix of one-time correlation
    g2 = np.zeros((tot_channels, num_qs+1), dtype=np.float64)

    num_pixels = np.delete(num_pixels, [0, num_qs+1])
    num_imgs = img_stack.shape[0]  # number of images(frames)

    for i in range(2, num_imgs):

        # delay times for each image
        delay_nums = [x for x in (i - np.array(lag_steps)) if x > 0]
        # number of terms for averaging
        num_terms = num_imgs - len(delay_nums)

        # buffer numbers for each correlation
        buf_nums = [x for x in (i - np.array(lag_steps)) if x > 0]
        buf_nums.pop()
        buf_nums.insert(0, i)

        # updating future intensities
        IF = img_stack[buf_nums]
        # updating past intensities
        IP = img_stack[delay_nums]
        IFP = (IF*IP)

        for j in range (0, len(delay_nums)):
            GD[j, :] = np.bincount(q_inds, weights=np.ravel(IFP[j, :]))
            G_del[j, :] = np.delete(GD[j, :], [0, num_qs + 1])
            IAFD[j, :] = np.bincount(q_inds, weights=np.ravel(IF[j, :]))
            IAF_del[j, :] = np.delete(IAFD[j,:], [0, num_qs +1])
            IAPD[j, :] = np.bincount(q_inds, weights=np.ravel(IP[j, :]))
            IAP_del[j, :] = np.delete(IAPD[j, :], [0, num_qs + 1])

        G += (G_del/num_pixels - G)/(num_terms)
        IAF += (IAF_del/num_pixels - IAF)/(num_terms)
        IAP += (IAP_del/num_pixels - IAP)/(num_terms)

    g2 = G/(IAF*IAP)

    return G, IAF, IAP, g2


def plot_g2(g2):
    plt.title(" 1 time correlation plots ")
    for i in range(0, num_qs):
        plt.plot(g2[i] + (i*0.1))
    plt.show()
    return

if __name__ == "__main__":
    #  image data as a stack
    img_stack = np.load("img_stack5301_5401.npy")

    detector_size = (256, 256)
    pixel_size = (0.0135*8, 0.0135*8)
    calibrated_center = (143, 123)  # (mm)
    dist_sample = 2230  # (mm)
    wavelength = 1.546  # (Angstrom )

    first_q = 0.0016  # (1/ Angstrom)
    delta_q = 2e-4  # (1/ Angstrom)
    step_q = 2e-4  # (1/ Angstrom)
    tolerance = 0
    num_qs = 8  # number of Q rings
    lag_time = 0.00130068  # (s)

    num_levels = 3
    num_channels = 8

    # delta=40, theta=15, chi = 90, phi = 30, mu = 10.0, gamma=5.0
    setting_angles = np.array([0., 0., 0., 0., 0., 0.])

    # UB matrix (orientation matrix) 3x3 matrix
    ub_mat = np.identity(3)

    # hkl values for all the pixels
    hkl_val = recip.process_to_q(setting_angles, detector_size,
                                 pixel_size, calibrated_center,
                                 dist_sample, wavelength, ub_mat,
                                 frame_mode=None)

    # Q values, Q indices for all pixels and Q ring
    # edge values for each ring
    # and number of pixels for each Q ring
    q_values, q_inds, q_ring_val , \
    num_pixels = recip.q_data(hkl_val, num_qs, first_q,
                           step_q, delta_q, detector_size)

    plot_q_rings(q_inds, detector_size)

    G, IAP, IAF, g2 = one_time_corr(num_levels, num_channels,
                                    num_qs, img_stack, q_inds, num_pixels, lag_time)
    plot_g2(g2.T)
