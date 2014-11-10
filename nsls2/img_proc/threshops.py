# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Oct. 2013
"""
This module contains tools for thresholding data sets.
"""

"""
REVISION LOG: (FORMAT: "PROGRAMMER INITIALS: DATE -- RECORD")
GCI: 2/11/2014 -- Modifying documentation of the package functions for 
    inclusion in the bulk module pull to GITHUB
GCI: 2/19/2014 -- Renaming module from C4_threshops to threshops.py
    Updating documentation to docstring convention.
    Changed module back from class structure to a simple module containing
    thresholding function definitions.
GCI: 9/23/14 -- Modified all functions to auto-wrap into the image
    processing library in vistrails.
"""

import numpy as np
import skimage.filter as sk


def thresh_globalGT(src_data, thresh_value, md_dict=None):
    """
    This function is a basic thresholding operation allowing for isolation, or 
    selection, of all voxels having an intensity value greater than the 
    specificed threshold value. The result is a binary array of equal size 
    containing voxel values of either 0 or 1.

    Parameters
    ----------
    src_data : ndarray
        Specify the volume to be thresholded
    
    thresh_value : float
        Specify the threshold value, above which contains the desired voxel 
        values to be isolated. Input value is of type float, however,
        the dtype of this parameter will be automatically converted to the
        dtype of the source data if the source data is not of type float.

    md_dict : dict, optional
        Metadata dictionary for the data set(s) being segmented or analyzed,
        or for the analysis being conducted

    Returns
    -------
    output : ndarray
        The function returns a binary array where all voxels with values equal 
        to 1 correspond to voxels within the identified threshold range.
    """

    if thresh_value > np.amax(src_data):
        raise ValueError("Selected threshold value is greater than the "
                         "maximum array value. Current settings will "
                         "result in an empty array. Current "
                         "thresh_value = {0}, array maximum = {1}, "
                         "resulting array fill value = {2}".format(
                         thresh_value, np.amax(src_data), False))
    elif thresh_value < np.amin(src_data):
        raise ValueError("Selected threshold value is less than the "
                         "minimum array value. Current settings will "
                         "result in an filled array. Current "
                         "thresh_value = {0}, array maximum = {1}, "
                         "resulting array fill value = {2}".format(
                         thresh_value, np.amin(src_data), True))
    output = (src_data > thresh_value)
    return output


def thresh_globalLT(src_data, thresh_value):
    """
    This function is a basic thresholding operation allowing for isolation, or 
    selection, of all voxels having an intensity value less than the
    specificed
    threshold value. The result is a binary array of equal size containing 
    voxel values of either 0 or 1.

    Parameters
    ----------
    src_data : ndarray
        Specify the volume to be thresholded

    thresh_value : float
        Specify the threshold value, below which contains the desired voxel 
        values to be isolated.

    Returns
    -------
    output : ndarray
        The function returns a binary array where all voxels with values equal 
        to 1 correspond to voxels within the identified threshold range.
    """
    if thresh_value > np.amax(src_data):
        raise ValueError("Selected threshold value is greater than the "
                         "maximum array value. Current settings will "
                         "result in an filled array. Current "
                         "thresh_value = {0}, array maximum = {1}, "
                         "resulting array fill value = {2}".format(
                         thresh_value, np.amax(src_data), True))
    elif thresh_value < np.amin(src_data):
        raise ValueError("Selected threshold value is less than the "
                         "minimum array value. Current settings will "
                         "result in an empty array. Current "
                         "thresh_value = {0}, array maximum = {1}, "
                         "resulting array fill value = {2}".format(
                         thresh_value, np.amin(src_data), False))
    output = (src_data < thresh_value)
    return output


def thresh_bounded(src_data, thresh_value_min, thresh_value_max):
    """
    This function is a basic thresholding operation allowing for isolation, or 
    selection, of all voxels having an intensity value between the specificed 
    threshold value boundaries. The result is a binary array of equal size 
    containing voxel values of either 0 or 1.

    Parameters
    ----------
    src_data : ndarray
        Specify the volume to be thresholded.

    thresh_value_MIN : float
        Specify the lower threshold boundary.

    thresh_value_MAX : float
        Specify the upper threshold boundary.

    Returns
    -------
    output : ndarray
        The function returns a binary array where all voxels with values equal 
        to 1 correspond to voxels within the identified threshold range.
    """
    # Check to make sure that maximum threshold boundary is less than or
    # equal to the maximum array value, and
    if thresh_value_max > np.amax(src_data):
        thresh_value_max = np.amax(src_data)
    elif thresh_value_min < np.amin(src_data):
        thresh_value_min = np.amin(src_data)

    if thresh_value_min > np.amax(src_data):
        raise ValueError("Selected threshold value is greater than the "
                         "maximum array value. Current settings will "
                         "result in an filled array. Current "
                         "thresh_value = {0}, array maximum = {1}, "
                         "resulting array fill value = {2}".format(
                         thresh_value_min, np.amax(src_data), True))
    elif thresh_value_max < np.amin(src_data):
        raise ValueError("Selected threshold value is less than the "
                         "minimum array value. Current settings will "
                         "result in an empty array. Current "
                         "thresh_value = {0}, array maximum = {1}, "
                         "resulting array fill value = {2}".format(
                         thresh_value, np.amin(src_data), False))
    vox_abv_min_value = (thresh_value_min <= src_data)
    vox_abv_max_value = (thresh_value_max <= src_data)
    output = vox_abv_min_value - vox_abv_max_value
    return output


def thresh_adapt(src_data, kernel_size, filter_type='gaussian'):
    """
    Applies an adaptive threshold to the source data set.

    Parameters
    ----------
    src_data : ndarray
        Source data on which to apply the threshold algorithm

    kernel_size : integer
        Specify kernel size for automatic thresholding operation
        Note: Value must be an odd valued integer

    filter_type : string
        Filter type options:
            method : {'generic', 'gaussian', 'mean', 'median'}, optional
            Method used to determine adaptive threshold for local
            neighbourhood in weighted mean image.
            * 'generic': use custom function (see `param` parameter)
            * 'gaussian': apply gaussian filter (see `param` parameter
                for custom sigma value)
            * 'mean': apply arithmetic mean filter
            * 'median': apply median rank filter

    Returns
    -------
    output_data : ndarray
        The function returns a binary array where all voxels with values equal
        to 1 correspond to voxels within the identified threshold range.
    """

    if type(kernel_size) != int:
        raise TypeError('Specified value for kernel_size is not an integer!')
    if (kernel_size % 2) == 0:
        raise ValueError('Specified kernel_size value is not an odd valued '
                         'integer!')
    output_data = sk.threshold_adaptive(src_data, kernel_size, filter_type,
                                        offset=0, param=None)
    return output_data


def thresh_otsu(src_data):
    """
    This function automatically determines a threshold value for the source
    data set using Otsu's method. Both the determined threshold
    value and the thresholded binary data set are returned as outputs

    Parameters
    ----------
    src_data : ndarray
        Source data on which to apply the threshold algorithm

    Returns
    -------
    output : ndarray
        The function returns a binary array where all voxels with values equal
        to 1 correspond to voxels within the identified threshold range.

    thresh_value : float
        The threshold value determined by the thresholding function and used
        to generate the binary volume data set
    """
    data_max = np.amax(src_data)
    data_min = np.amin(src_data)
    if (data_max - data_min) > 1000:
        num_bins = (data_max - data_min) / 2
    else:
        num_bins = (data_max - data_min)
    thresh_value = sk.threshold_otsu(src_data, num_bins)
    output_data = thresh_globalGT(src_data, thresh_value)
    return output_data, thresh_value


def thresh_yen(src_data):
    """
    This function automatically determines a threshold value for the source
    data set using Yen's method. Both the determined threshold
    value and the thresholded binary data set are returned as outputs

    Parameters
    ----------
    src_data : ndarray
        Source data on which to apply the threshold algorithm

    Returns
    -------
    output : ndarray
        The function returns a binary array where all voxels with values equal
        to 1 correspond to voxels within the identified threshold range.

    thresh_value : float
        The threshold value determined by the thresholding function and used
        to generate the binary volume data set
    """
    data_max = np.amax(src_data)
    data_min = np.amin(src_data)
    if (data_max - data_min) > 1000:
        num_bins = (data_max - data_min) / 2
    else:
        num_bins = (data_max - data_min)
    thresh_value = sk.threshold_otsu(src_data, num_bins)
    output_data = thresh_globalGT(src_data, thresh_value)
    return output_data, thresh_value


def thresh_isodata(src_data):
    """
    This function automatically determines a threshold value for the source
    data set using the Ridler-Calvard method. Both the determined threshold
    value and the thresholded binary data set are returned as outputs

    Parameters
    ----------
    src_data : ndarray
        Source data on which to apply the threshold algorithm

    Returns
    -------
    output_data : ndarray
        The function returns a binary array where all voxels with values equal
        to 1 correspond to voxels within the identified threshold range.

    thresh_value : float
        The threshold value determined by the thresholding function and used
        to generate the binary volume data set
    """
    data_max = np.amax(src_data)
    data_min = np.amin(src_data)
    if (data_max - data_min) > 1000:
        num_bins = int((data_max - data_min) / 2)
    else:
        num_bins = int((data_max - data_min))
    thresh_value = sk.threshold_isodata(src_data, num_bins)
    output_data = thresh_globalGT(src_data, thresh_value)
    return output_data, thresh_value


thresh_adapt.k_shape = ['2D', '3D']
thresh_adapt.filter_type = ['generic', 'gaussian', 'mean', 'median']
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# TODO Add simple multi thresholding tool
# TODO Add auto thresholding methods including:
# 1st derivative
# entropy
# moments
# factorization
# OTHERS...
# MULTI-THRESHOLDING
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# def thresh_multi(self, src_data):
# output = not src_data
# print 'Operation successful: ' + "NOT " + src_data
# return output
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# FIRST DERIVATIVE THRESHOLDING
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# def thresh_firstDerivative(self, op_type='offset', offset_image=0):
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# k_shape : string
# Specify whether the kernel to be applied during automatic thresholding
# is to be 2-dimensional ('2D') or 3-dimensional ('3D')
# k_size : integer
# Specify the size (in pixels) of the kernel to be used for thresholding
# Note: specified kernel sizes must be odd integers greater than 1

