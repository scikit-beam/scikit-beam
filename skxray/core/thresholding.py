# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Oct. 2013
"""
This module contains tools for thresholding data sets.
"""


import numpy as np
import skimage.filter as sk

#def check_saturation (src_data):
#    """
#    This function is a helper function to test for potentially erroneous
#    thresholding results.
#    First:
#        The function checks the entire source data set to confirm that the
#        binary result is not completely "saturated" with all values set
#        equal to 1, nor that the source array is "empty" with all values set
#        equal to 0.
#        If the source array is either completely saturated or empty, then a
#        warning is produced to alert the user of a potential error.
#    Second:
#        If the source array has more than 2 dimensions, then the array is
#        evaluated slice-by-slice along each of the axial dimensions. Slices
#        that meet the evaluation criteria, with all values eq. 0 (empty) or
#        eq. 1 (saturated) are logged to assist the user in identifying:
#            (A) exterior regions outside of the Region Of Interest (ROI), for
#                cropping, or
#            (B) artifacts or problematic regions,
#            (C) the slice record could also be useful for quickly
#                identifying ROI for additional analysis.
#    Parameters
#    __________
#    src_data : array
#        Binary array, typically the result of thresholding
#    Returns
#    """#
#
#    z_dim, y_dim, x_dim = src_data.shape
#    if np.all(src_data):
#        raise ValueError("Binary array is saturated (all cells eval to
# True).")

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
    thresh_value_min : float
        Specify the lower threshold boundary.
    thresh_value_max : float
        Specify the upper threshold boundary.
    Returns
    -------
    output : ndarray
        The function returns a binary array where all voxels with values equal
        to 1 correspond to voxels within the identified threshold range.
    """
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
    if (np.ptp(src_data)) > 1000:
        num_bins = int(np.ptp(src_data) / 2)
    else:
        num_bins = int(np.ptp(src_data))
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
    if (np.ptp(src_data)) > 1000:
        num_bins = int(np.ptp(src_data) / 2)
    else:
        num_bins = int(np.ptp(src_data))
    thresh_value = sk.threshold_yen(src_data, num_bins)
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
    if (np.ptp(src_data)) > 1000:
        num_bins = int(np.ptp(src_data) / 2)
    else:
        num_bins = int(np.ptp(src_data))
    thresh_value = sk.threshold_isodata(src_data, num_bins)
    output_data = thresh_globalGT(src_data, thresh_value)
    return output_data, thresh_value