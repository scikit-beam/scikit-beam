# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Oct. 2013
"""
This module contains tools for thresholding data sets.
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

