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


def squared_dist_2D(dims):
    """
    Create array with pixel value equals squared euclidian distance
    from array center in 2D.

    Parameters
    ----------
    dims : list or tuple
        shape of the data

    Returns
    -------
    array :
        2D array to meet requirement
    """
    x_sq = (np.arange(dims[0]) - dims[0]/2)**2
    y_sq = (np.arange(dims[1]) - dims[1]/2)**2
    return x_sq.reshape([dims[0], 1]) + y_sq


def dist(dims):
    """
    Create array with pixel value equals euclidian distance from array center.

    Parameters
    ----------
    dims : list or tuple
        shape of the data

    Returns
    -------
    array :
        2D or 3D array
    """
    if np.size(dims) == 2:
        return np.sqrt(squared_dist_2D(dims))

    if np.size(dims) == 3:
        temp = squared_dist_2D(dims[:-1])
        z_sq = (np.arange(dims[2]) - dims[2]/2)**2
        return np.sqrt(temp.reshape([dims[0], dims[1], 1])
                       + z_sq.reshape([1, 1, dims[2]]))


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
        2D or 3D gaussian
    """
    x = dist(dims)
    y = np.exp(-(x / sigma)**2/2.)
    return y/np.sum(y)

