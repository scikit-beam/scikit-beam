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
    This module will provide statistical analysis for the speckle patterns
    
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
from scipy.ndimage.measurements import maximum, mean

try:
    iteritems = dict.iteritems
except AttributeError:
    iteritems = dict.items  # python 3

import logging
logger = logging.getLogger(__name__)


def roi_max_counts(images_sets, label_array):
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
            max_cts = max(max_cts, maximum(img, label_array))
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

    return {n: image[labels == n] for n in range(1, np.max(labels)+1)}


def time_series(number=2, number_of_images=50):
    """
    This will provide the geometric series for the integration.
    Last values of the series has to be less than or equal to number
    of images
    ex: number_of_images = 100
    number = 2, time_series =  1, 2, 4, 8, 16, 32, 64
    number = 3, time_series =  1, 3, 9, 27, 81

    Parameters
    ----------
    number : int, optional
        time steps for the integration

    number_of_images : int, optional
        number of images

    Return
    ------
    time_series : list
        time binning

    Note
    ----
    :math ::
     a + ar + ar^2 + ar^3 + ar^4 + ...

     a - first term in the series
     r - is the common ratio
    """

    time_series = [1]

    while time_series[-1]*number < number_of_images:
        time_series.append(time_series[-1]*number)
    return time_series


def mean_intensity_sets(images_set, labels):
    """
    Mean intensities for ROIS' of the labeled array for different image sets

    Parameters
    ----------
    images : array
        iterable of 4D arrays
        shapes is: (len(images_sets), )

    labels : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).

    Returns
    -------
    mean_int_labels : dict
        average intensity of each ROI as a dictionary
        shape len(images_sets)
        eg: 2 image sets,
        {image set 1 : (len(images in image set 1), number of labels),
        image set 2 : (len(images in image set 2), number of labels)}

    """
    return {n+1: mean_intensity(images_set[n],
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
    mean_intensity : array
        mean intensity of each ROI for the set of images as an array
        shape (len(images), number of labels)

    """
    if labels.shape != images[0].shape[0:]:
        raise ValueError("Shape of the images should be equal to"
                         " shape of the label array")

    index = np.unique(labels)[1:]
    mean_intensity = np.zeros((images.shape[0], index.shape[0]))

    for n, img in enumerate(images):
        mean_intensity[n] = mean(img, labels, index=index)

    return mean_intensity


def combine_mean_intensity(mean_int_dict):
    """
    Combine mean intensities of the images(all images sets) for each ROI

    Parameters
    ----------
    mean_int_dict : dict
        mean intensity of each ROI as a dictionary
        eg: 2 image sets,
        {image set 1 : (len(images in image set 1), number of labels),
        image set 2 : (len(images in image set 2), number of labels)}

    Returns
    -------
    combine_mean_int : array
        combine mean intensities of image sets for each ROI of labeled array
        shape (len(images in all image sets), number of labels)

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

    bin_centers = core.bin_edges_to_centers(bin_edges)[th_mask]

    return bin_centers, ring_averages
