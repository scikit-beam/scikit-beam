# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Developed at the NSLS-II, Brookhaven National Laboratory             #
# Developed by Sameera K. Abeykoon, January 2105                       #
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
This module is for functions specific to mask or threshold an image
basically to clean images
"""

from __future__ import absolute_import, division, print_function
import numpy as np

import logging
logger = logging.getLogger(__name__)


def bad_to_nan_gen(image_gen, bad_list):
    """
    This generator will convert the bad image array in the images into
    NAN(Not-A-Number) array

    Parameters
    ----------
    image_gen : array
        image_iterable : iterable of 2D arrays
     : list
        bad images list

    Yields
    ------
    img : array
        if image is bad it will convert to np.nan array otherwise no
        change to the array
    """
    for n, im in enumerate(image_gen):
        if n in bad_list:
            yield np.nan*np.ones_like(im)
        else:
            yield im


def threshold_mask(images, threshold, ther_mask=None):
    """
    This generator will create a threshold mask for images

    Parameters
    ----------
    images : array
        image_iterable : iterable of 2D arrays
    threshold: float
        threshold value to remove the hot spots in the image

    Yields
    -------
    thre_mask : array
        image mask to remove the hot spots
    """
    if ther_mask is None:
        thre_mask = np.ones_like(images[0])
    for im in images:
        bad_pixels = np.where(im >= threshold)
        if len(bad_pixels[0])!=0:
            thre_mask[bad_pixels] = 0
        yield thre_mask
