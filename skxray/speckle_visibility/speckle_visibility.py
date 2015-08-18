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

from six.moves import zip
from six import string_types


import skxray.correlation as corr
import skxray.roi as roi
import skxray.core as core
import scipy.ndimage.measurements as meas
# TODO  check this in skimage

try:
    iteritems = dict.iteritems
except AttributeError:
    iteritems = dict.items  # python 3

import logging
logger = logging.getLogger(__name__)

import numpy as np


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
        for img in img_set:
            max_cts = max(max_cts, meas.maximum(img, label_array))
    return max_cts


def roi_pixel_values(image, labels):
    """
    This will provide intensities of the ROI's of the labeled array
    according to the pixel list
    eg: intensities of the rings of the labeled array

    Parameters
    ----------
    image : array
        image data dimensions are: (rr, cc)

    labels : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).

    Returns
    -------
    roi_pix : dict
        intensities of the ROI's of the labeled array according
        to the pixel list
        {ROI 1 : intensities of the pixels of ROI 1, ROI 2 : intensities of
         the pixels of ROI 2}
    """

    if labels.shape != image.shape:
        raise ValueError("Shape of the image data should be equal to"
                         " shape of the labeled array")

    #labels, indices = corr.extract_label_indices(label_array)
    label_num = np.unique(labels)[1:]

    #intensity_distribution = {}

    for n in label_num:
        value = (np.ravel(image_array)[indices[labels==n].tolist()])
        intensity_distribution[n] = value

    return {n: image[labels == n] for n in range(1, np.max(labels))}



def static_test_sets_one_label(sample_dict, label_array, num=1):
    """
    This will process the averaged intensity for the required ROI for different
    data sets (dictionary for different data sets)
    eg: ring averaged intensity for the required labeled ring for different
    image data sets.

    Parameters
    ----------
    sample_dict : dict:

    label_array : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).

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

    average_intensity_sets = {}

    for key, img in dict(sample_dict).iteritems():
        average_intensity_sets[key] = static_tests_one_label(img, label_array,
                                                             num)
    return average_intensity_sets



def static_test_sets(sample_dict, label_array):
    """
    This will process the averaged intensity for the required ROI's for different
    data sets (dictionary for different data sets)
    eg: ring averaged intensity for the required ROI's for different
    image data sets.

    Parameters
    ----------
    images : array
        iterable of 2D arrays
        dimensions are: (rr, cc)

    labels : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).

    Returns
    -------
    mean_int_labels : dict
        average intensity of each ROI as a dictionary
        {roi 1: average intensities, roi 2 : average intensities}

    """
    return {n+1 : mean_intensity(images_set[n],
                                 labels) for n in range(len(images_set))}


def mean_intensity(images, labels):
    """
    Mean intensities for ROIS' of the labeled array for set of images

    Parameters
    ----------
    images : array
        iterable of 2D arrays
        dimensions are: (rr, cc)

    labels : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).

    Returns
    -------
    mean_int : array
        mean intensity of each ROI for the set of images as an array
        shape (number of images in the set, number of labels)

    """
    if labels.shape != images[0].shape[0:]:
        raise ValueError("Shape of the images should be equal to"
                         " shape of the label array")

    index = np.unique(labels)[1: ]
    mean_int = np.zeros((images.shape[0], index.shape[0]))

    for n in range(images.shape[0]):
        mean_int[n] = meas.mean(images[n], labels, index=index)

    return mean_int


def combine_mean_intensity(mean_int_dict):
    """
    Parameters
    ----------
    mean_int_dict : dict
        mean intensity of each ROI as a dictionary
        {roi 1: average intensities, roi 2 : average intensities}
    """
    average_intensity = {}
    num = np.unique(label_array)[1:]

    for i in num:
        average_roi = static_tests_one_label(images, label_array, num=i+1)
        average_intensity[i] = average_roi

    return average_intensity


def time_bining(number=2, number_of_images=50):
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
    time_bin = [1]
    while time_bin[-1]*number<number_of_images:
        time_bin.append(time_bin[-1]*number)
    return time_bin

    edges = roi.ring_edges(inner_radius, width, num_rings=1)

    m_value = float('inf')
    for x in xrange(est_center[0]-var, est_center[0]+var):
        for y in xrange(est_center[1]-var, est_center[1]+var):
            rings = roi.rings(edges, (x, y), image.shape)
            if mask is not None:
                if mask.shape != image.shape:
                    raise ValueError("Shape of the mask should be equal to"
                         " shape of the image")
                else:
                    rings = rings*mask
            labels, indices = corr.extract_label_indices(rings)
            intensity_dist = intensity_distribution(image, rings)

            a = np.vstack([indices, np.ones(len(indices))]).T

            m, c = np.linalg.lstsq(a, intensity_dist.values()[0])[0]
            if m < m_value:
                m_value = m
                center = (x, y)

    return center


def circular_average(image, calibrated_center, thershold=0, nx=100,
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

    thershold : float, optional
        threshold value to mask

    nx : int, optional
        number of bins

    pixel_size : tuple, optional
        The size of a pixel in real units. (height, width). (mm)

    Returns
    -------
    bin_centers : array
        bin centers from bin edges

    ring_averages : array
        circular integration of SAXS intensity
    """
    radial_val = core.radial_grid(calibrated_center, image.shape,
                                  pixel_size)

    bin_edges, sums, counts = core.bin_1D(np.ravel(radial_val),
                                          np.ravel(image), nx)
    th_mask = counts > thershold
    ring_average = sums[th_mask] / counts[th_mask]

    bin_centers = core.bin_edges_to_centers(bin_edges)

    return bin_centers, ring_average


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
    combine_mean_int : array
        combine mean intensities of image sets for each ROI of labeled array

    """
    return np.vstack(list(mean_int_dict.values()))


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
