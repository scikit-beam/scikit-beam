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

import numpy as np
from scipy.signal import fftconvolve


def sgolay2d(image, window_size, order, derivative=None):
    """
    Savitzky-Golay filter for 2D image arrays.
    See: http://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html

    Parameters
    ----------
    image : ndarray, shape (N,M)
        image to be smoothed.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    smooth_image : ndarray, shape (N,M)
        the smoothed image .
    """
    # number of terms in the polynomial expression
    n_terms = (order + 1) * (order + 2) / 2.0

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    if window_size**2 < n_terms:
        raise ValueError("order is too high for the window size")

    half_size = window_size // 2

    # exponents of the polynomial.
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [(k - n, n) for k in range(order + 1) for n in range(k + 1)]

    # coordinates of points
    ind = np.arange(-half_size, half_size + 1, dtype=np.float64)
    dx = np.repeat(ind, window_size)
    dy = np.tile(ind, [window_size, 1]).reshape(
        window_size**2,
    )

    # build matrix of system of equation
    A = np.empty((window_size**2, len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (dx ** exp[0]) * (dy ** exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = image.shape[0] + 2 * half_size, image.shape[1] + 2 * half_size
    smooth_image = np.zeros((new_shape))
    # top band
    band = image[0, :]
    smooth_image[:half_size, half_size:-half_size] = band - np.abs(np.flipud(image[1 : half_size + 1, :]) - band)
    # bottom band
    band = image[-1, :]
    smooth_image[-half_size:, half_size:-half_size] = band + np.abs(
        np.flipud(image[-half_size - 1 : -1, :]) - band
    )
    # left band
    band = np.tile(image[:, 0].reshape(-1, 1), [1, half_size])
    smooth_image[half_size:-half_size, :half_size] = band - np.abs(np.fliplr(image[:, 1 : half_size + 1]) - band)
    # right band
    band = np.tile(image[:, -1].reshape(-1, 1), [1, half_size])
    smooth_image[half_size:-half_size, -half_size:] = band + np.abs(
        np.fliplr(image[:, -half_size - 1 : -1]) - band
    )
    # central band
    smooth_image[half_size:-half_size, half_size:-half_size] = image

    # top left corner
    band = image[0, 0]
    smooth_image[:half_size, :half_size] = band - np.abs(
        np.flipud(np.fliplr(image[1 : half_size + 1, 1 : half_size + 1])) - band
    )
    # bottom right corner
    band = image[-1, -1]
    smooth_image[-half_size:, -half_size:] = band + np.abs(
        np.flipud(np.fliplr(image[-half_size - 1 : -1, -half_size - 1 : -1])) - band
    )

    # top right corner
    band = smooth_image[half_size, -half_size:]
    smooth_image[:half_size, -half_size:] = band - np.abs(
        np.flipud(smooth_image[half_size + 1 : 2 * half_size + 1, -half_size:]) - band
    )
    # bottom left corner
    band = smooth_image[-half_size:, half_size].reshape(-1, 1)
    smooth_image[-half_size:, :half_size] = band - np.abs(
        np.fliplr(smooth_image[-half_size:, half_size + 1 : 2 * half_size + 1]) - band
    )

    # solve system and convolve
    if derivative is None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return fftconvolve(smooth_image, m, mode="valid")
    elif derivative == "col":
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return fftconvolve(smooth_image, -c, mode="valid")
    elif derivative == "row":
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return fftconvolve(smooth_image, -r, mode="valid")
    elif derivative == "both":
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return fftconvolve(smooth_image, -r, mode="valid"), fftconvolve(smooth_image, -c, mode="valid")
