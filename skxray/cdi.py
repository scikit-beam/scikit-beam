# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# @author: Li Li (lili@bnl.gov)                                        #
# created on 03/27/2015                                                #
#                                                                      #
# Original code from Xiaojing Huang (xjhuang@bnl.gov) and Li Li        #
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


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np

import logging
logger = logging.getLogger(__name__)


def _dist(dims):
    """
    Create array with pixel value equals to the distance from array center.

    Parameters
    ----------
    dims : list or tuple
        shape of array to create

    Returns
    -------
    arr : np.ndarray
        ND array whose pixels are equal to the distance from the center
        of the array of shape `dims`
    """
    dist_sum = []
    shape = np.ones(len(dims))
    for idx, d in enumerate(dims):
        vec = (np.arange(d) - d // 2) ** 2
        shape[idx] = -1
        vec = vec.reshape(*shape)
        shape[idx] = 1
        dist_sum.append(vec)

    return np.sqrt(np.sum(dist_sum, axis=0))


def gauss(dims, sigma):
    """
    Generate Gaussian function in 2D or 3D.

    Parameters
    ----------
    dims : list or tuple
        shape of the data
    sigma : float
        standard deviation of gaussian function

    Returns
    -------
    Array :
        ND gaussian
    """
    x = _dist(dims)
    y = np.exp(-(x / sigma)**2 / 2)
    return y / np.sum(y)


def convolution(array1, array2):
    """
    Calculate convolution of two arrays. Transfer into q space to perform the calculation.

    Parameters
    ----------
    array1 : array
        The size of array1 needs to be normalized.
    array2 : array
        The size of array2 keeps the same

    Returns
    -------
    array :
        convolution result

    Notes
    -----
    Another option is to use scipy.signal.fftconvolve. Some differences between
    the scipy function and this function were found at the boundary.  See
    `this issue on github <https://github.com/Nikea/scikit-xray/issues/258>`_
    for details.
    """
    fft_norm = lambda x:  np.fft.fftshift(np.fft.fftn(x)) / np.sqrt(np.size(x))
    fft_1 = fft_norm(array1)
    fft_2 = fft_norm(array2)
    return np.abs(np.fft.ifftshift(np.fft.ifftn(fft_1*fft_2)) *
                  np.sqrt(np.size(array2)))

