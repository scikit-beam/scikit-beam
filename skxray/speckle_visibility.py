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
    This module will provide analysis codes for static tests for the image
    data and for the X-ray Speckle Visibility Spectroscopy (XSVS)bi
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six

from six.moves import zip
from six import string_types

import logging
logger = logging.getLogger(__name__)

import numpy as np
from time import time

from ConfigParser import RawConfigParser
from os.path import isfile
import os
from sys import argv, stdout
import sys

import skxray.correlation as corr


def intensity_distribution(image_array, label_array):
    """
    This will provide the intensity distribution of the ROI"s
    eg: radial intensity distributions of a
    rings of the label array
    Parameters
    ----------
    image_array : array
        image data dimensions are: (rr, cc)
    label_array : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).
    Returns
    -------
    radial_intensity : dict
        radial intensity of each ROI's
    """

    labels, indices = corr.extract_label_indices(label_array)
    label_num = np.unique(labels)

    intensity_distribution = {}

    for n in label_num:
        value = (np.ravel(image_array)[indices[labels==n].tolist()])
        intensity_distribution[n] = value

    return intensity_distribution


def static_test_sets(image_dict, label_array, num=1):
    """
    This will process the averaged intensity for the required ROI for different
    data sets (dictionary for different data sets)
    eg: ring averaged intensity for the required labeled ring for different
    image data sets.

    Parameters
    ----------
    image_dict : dict
    label_array : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).

    num : int, optional
        Required  ROI label

    Returns
    -------
    average_intensity : dict
    """

    average_intensity_sets = {}

    for key, img in dict(image_dict).iteritems():
        average_intensity_sets[key] = static_tests_one_label(img, label_array,
                                                             num)

    return average_intensity_sets


def static_tests_one_label(images, label_array, num=1):
    """
    This will provide the average intensity values and
    intensity values of one region of interests for the
    required intensity array of images.

    Parameters
    ----------
    images : array
        iterable of 2D arrays
        dimensions are: (rr, cc)

    label_array : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).

    num : int, 1
        Required  ROI label

    Returns
    -------
    average_intensity : array
        average intensity of ROI's
        for the intensity array of images
        dimensions are : [num_images][len(indices)]

    """
    labels, indices = corr.extract_label_indices(label_array)
    average_intensity = []

    for n, img in enumerate(images.operands[0]):
        value = (np.ravel(img)[indices[num].tolist()])
        average_intensity.append(np.mean(value))

    return average_intensity


def static_test(images, label_array):

    labels, indices = corr.extract_label_indices(label_array)

    average_intensity = {}
    num = np.unique(labels)[1:]

    for i in num:
        average_intensity = static_tests_one_label(images, label_array, num)


def static_tests(images, label_array):
    labels, indices = corr.extract_label_indices(label_array)

    average_intensity = {}
    for n, img in enumerate(images.operands[0]):
        value = np.ravel(img)[indices.tolist()]


def time_bin(number=2, number_of_images=50):
    """
    This will provide the time binning for the integration.

    Parameters
    ----------
    number : int, optional
        time steps for the integration
        ex:
        1, 2, 4, 8, 16, ...
        1, 3, 9, 27, ...

    number_of_images : int, 50
        number of images
        
    Return
    ------
    time_bin : list
        time bining
    """
    time_step = time_bin = [1]

    while time_step<number_of_images:
        time_step = time_bin[-1]*number
        time_bin.append(time_step)
    return time_bin





