# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Developed at the NSLS-II, Brookhaven National Laboratory             #
# Developed by Sameera K. Abeykoon, May 2015                           #
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
    This module will provide statistics analysis for the speckle pattern
    to use in X-ray Speckle Visibility Spectroscopy (XSVS)
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np
from six.moves import zip
from six import string_types

import skxray.correlation as corr
import skxray.roi as roi
import skxray.core as core

import scipy.ndimage as ndi

try:
    iteritems = dict.iteritems
except AttributeError:
    iteritems = dict.items  # python 3

import logging
logger = logging.getLogger(__name__)


def max_counts(images_sets, label_array):
    """
    This will determine the highest speckle counts occurred in the required
    ROI's in required images.

    Parameters
    ----------
    images_sets : array
        sets of images as an array

    label_array : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).

    Returns
    -------
    max_counts : int
        maximum speckle counts
    """
    max_cts = 0
    for img_set in images_sets:
        for n, img in enumerate(img_set.operands[0]):
            frame_max = ndi.measurements.maximum(img, label_array)
            max_cts = max(max_cts, frame_max)
    return max_cts


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

    if label_array.shape != image_array.shape:
        raise ValueError("Shape of the image data should be equal to"
                         " shape of the labeled array")

    labels, indices = corr.extract_label_indices(label_array)
    label_num = np.unique(labels)

    intensity_distribution = {}

    for n in label_num:
        value = (np.ravel(image_array)[indices[labels == n].tolist()])
        intensity_distribution[n] = value

    return intensity_distribution


def time_bining(number=2, number_of_images=50):
    """
    This will provide the geometric series for the integration.
    Last values of the series has to be less than or equal to number
    of images
    ex:
        1, 2, 4, 8, 16, ...
        1, 3, 9, 27, ...

    Parameters
    ----------
    number : int, optional
        time steps for the integration


    number_of_images : int, 50
        number of images

    Return
    ------
    time_bin : list
        time binning

    Note
    ----
    :math ::
     a + ar + ar^2 + ar^3 + ar^4 + ...

     a - first term in the series
     r - is the common ratio
    """

    time_bin = [1]

    while time_bin[-1]*number < number_of_images:
        time_bin.append(time_bin[-1]*number)
    return time_bin


def static_test_sets_one_label(sample_dict, label_array, num=1):
    """
    This will process the averaged intensity for the required ROI for different
    data sets (dictionary for different data sets)
    eg: ring averaged intensity for the required labeled ring for different
    image data sets.

    Parameters
    ----------
    sample_dict : dict
        image sets given as a dictionary

    label_array : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).

    num : int, optional
        Required  ROI label

    Returns
    -------
    average_intensity : dict
        average intensity of one ROI
        for the intensity array of image sets

    combine_averages : array
        combine intensity averages of one ROI for sets of images
    """

    average_intensity_sets = {}

    for key, img in iteritems(sample_dict):
        average_intensity_sets[key] = static_tests_one_label(img, label_array,
                                                             num)

    combine_averages = np.concatenate(list(average_intensity_sets.values()))

    return average_intensity_sets, combine_averages


def static_tests_one_label(images, label_array, num=1):
    """
    This will provide the average intensity values and intensity values of
    one ROI for the required intensity array of images.

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
    if label_array.shape != images.operands[0].shape[1:]:
        raise ValueError("Shape of the images should be equal to"
                         " shape of the label array")

    labels, indices = corr.extract_label_indices(label_array)
    average_intensity = []

    for n, img in enumerate(images.operands[0]):
        value = (np.ravel(img)[indices[num].tolist()])
        average_intensity.append(np.mean(value))

    return average_intensity


def static_test(images, label_array):
    """
    Averaged intensities for ROIS'

    Parameters
    ----------
    images : array
        iterable of 2D arrays
        dimensions are: (rr, cc)

    label_array : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).

    Returns
    -------
    average_intensity : dict
        average intensity of each ROI as a dictionary
        {roi 1: average intensities, roi 2 : average intensities}

    """
    average_intensity = {}
    num = np.unique(label_array)[1:]

    for i in num:
        average_roi = static_tests_one_label(images, label_array, num=i+1)
        average_intensity[i] = average_roi

    return average_intensity


def static_test_sets(sample_dict, label_array):
    """
    This will process the averaged intensity for the required ROI's for
    different data sets (dictionary for different data sets)
    eg: ring averaged intensity for the required ROI's for different
    image data sets.

    Parameters
    ----------
    sample_dict : dict

    label_array : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).

    num : int, optional
        Required  ROI label

    Returns
    -------
    average_intensity : dict
        average intensity of each image sets and for each ROI's
        eg:
        {image_set1: {roi_1: average intensities, roi_2: average intensities},
         image_set2: {roi_1: average intensities, roi_2: average intensities}}
    """

    average_intensity_sets = {}

    for key, img in iteritems(sample_dict):
        average_intensity_sets[key] = static_test(img, label_array)
    return average_intensity_sets


def circular_average(image, calibrated_center, threshold=0, nx=100,
                     pixel_size=None):
    """
    Circular average(radial integration) of the intensity distribution of
    the image data.

    Parameters
    ----------
    image : array
        input image

    calibrated_center : tuple
        The center in pixels-units (row, col)

    threshold : int, optional
        threshold value to mask

    nx : int, optional
        number of bins

    pixel_size : tuple, optional
        The size of a pixel in real units. (height, width). (mm)

    Returns
    -------
    bin_centers : array
        bin centers from bin edges
        shape [nx]

    ring_averages : array
        circular integration of intensity
    """
    radial_val = core.radial_grid(calibrated_center, image.shape,
                                  pixel_size)

    bin_edges, sums, counts = core.bin_1D(np.ravel(radial_val),
                                          np.ravel(image), nx)
    th_mask = counts > threshold
    ring_averages = sums[th_mask] / counts[th_mask]

    bin_centers = core.bin_edges_to_centers(bin_edges)

    return bin_centers, ring_averages
