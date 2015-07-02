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
    Return the brightest pixel in any ROI in any image in the image set.

    Parameters
    ----------
    images_sets : array
        iterable of 4D arrays
        shapes is: (len(images_sets), )

    label_array : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).

    Returns
    -------
    max_counts : int
        maximum pixel counts
    """
    max_cts = 0
    for img_set in images_sets:
        for img in img_set:
            max_cts = max(max_cts, maximum(img, label_array))
    return max_cts


def roi_pixel_values(image, labels, index=None):
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

    index_list : list, optional
        labels list
        eg: 5 ROI's
        index = [1, 2, 3, 4, 5]

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
    if index is None:
        index = np.arange(1, np.max(labels) + 1)

    roi_pix = {n: image[labels == n] for n in index}
    return roi_pix, index


def mean_intensity_sets(images_set, labels):
    """
    Mean intensities for ROIS' of the labeled array for different image sets

    Parameters
    ----------
    images_set : array
        images sets
        shapes is: (len(images_sets), )
        one images_set is iterable of 2D arrays dimensions are: (rr, cc)

    labels : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).

    Returns
    -------
    mean_intensity_dict : dict
        average intensity of each ROI as a dictionary
        shape len(images_sets)
        eg: 2 image sets,
        {image set 1 : len(images in image set 1),
        image set 2 : len(images in image set 2)}

    index_list : list
        labels list for each image set

    """
    mean_intensity_dict = {}
    index_list = []
    for n in range(len(images_set)):
        mean_int, index = mean_intensity(images_set[n], labels)
        mean_intensity_dict[n] = mean_int
        index_list.append(index)

    return mean_intensity_dict, index_list


def mean_intensity(images, labels, index=None):
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

    index : list
        labels list
        eg: 5 ROI's
        index = [1, 2, 3, 4, 5]

    Returns
    -------
    mean_intensity : array
        mean intensity of each ROI for the set of images as an array
        shape (len(images), number of labels)

    """
    if labels.shape != images[0].shape[0:]:
        raise ValueError("Shape of the images should be equal to"
                         " shape of the label array")
    if index is None:
        index = np.arange(1, np.max(labels) + 1)

    mean_intensity = np.zeros((images.shape[0], index.shape[0]))
    for n, img in enumerate(images):
        mean_intensity[n] = mean(img, labels, index=index)

    return mean_intensity, index


def combine_mean_intensity(mean_int_dict, index_list):
    """
    Combine mean intensities of the images(all images sets) for each ROI
    if the labels list of all the images are same

    Parameters
    ----------
    mean_int_dict : dict
        mean intensity of each ROI as a dictionary
        eg: 2 image sets,
        {image set 1 : (len(images in image set 1),
        image set 2 : (len(images in image set 2)}

    index_list : list
        labels list for each image sets

    Returns
    -------
    combine_mean_int : array
        combine mean intensities of image sets for each ROI of labeled array
        shape (len(images in all image sets), number of labels)

    """
    if np.all(map(lambda x: x == index_list[0], index_list)):
        combine_mean_intensity = np.vstack(list(mean_int_dict.values()))
    else:
        raise ValueError("Labels list for the image sets are different")

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


def roi_kymograph(images, labels, num):
    """
    This function will provide data for graphical representation of pixels
    variation over time for required ROI.

    Parameters
    ----------
    images : array
        iterable of 2D arrays
        dimensions are: (rr, cc)

    labels : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).

    num : int
        required ROI label

    Returns
    -------
    roi_kymograph : array

    """
    roi_kymo = []
    for n, img in enumerate(images):
        roi_kymo.append(list(roi_pixel_values(img,
                                              labels == num)[0].values())[0])

    return np.matrix(roi_kymo)
