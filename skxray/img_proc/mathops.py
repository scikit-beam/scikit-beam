# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Oct. 2013
"""
# This class is designed to facilitate image arithmetic and logical operations
# on image data sets.
"""

import numpy as np


def add(src_data,
        offset_data):
    """
    This function enables the addition of EITHER two images or volume 
    data sets, OR an image/data set and a value. This function is typically
    used for offset purposes, or basic recombination of several isolated 
    materials or phases into a single segmented volume.

    Parameters
    ----------
    src_data : array
        Specifies the data set to be manipulated

    offset_data : {array, int, float}
        Specifies the data set or value withwhich to offset src_data_1

    Returns
    -------
    output : array
        Returns the resulting array to the designated variable

    Example
    -------
    result = mathops.add(img_1, img_2)
    """
    output = src_data + offset_data
    return output


def subtract(src_data,
             offset_data):
    """
    This function enables the subtraction of EITHER one image or volume data 
    set from another, OR reduction of all values in an image/data set by a set 
    value. This function is typically used for offset purposes, or basic 
    isolation of objects or materials/phases in a data set.

    Parameters
    ----------
    src_data : array
        Specifies the data set to be manipulated

    offset_data : {array, int, float}
        Specifies the data set or value withwhich to offset src_data_1

    Returns
    -------
    output : array
        Returns the resulting array to the designated variable

    Example
    -------
    result = mathops.subtract(img_1, img_2)
    NOTE:
    IF The volume being evaluated is an 8-bit int data set (min=0, max=255) 
    If the subtraction value is greater than any     original cell value, then 
    the operation circles around to the maximum value of 255 and starts counting
    down.
    For Example:
        Original cell value Image1[100,100] = 2
        Subtraction offset = 5
        $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        Operation Results:
            Original value = 2
            Final Value = 253
        Because:
            2 (- 0) = 2
            2 (- 1) = 1
            2 (- 2) = 0
            2 (- 3) = 255
            2 (- 4) = 254
            2 (- 5) = 253
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    output = src_data - offset_data
    return output


def multiply(src_data,
             offset_data):
    """
    This function allows the multiplication of EITHER one image or volume 
    data set with another, OR multiplication of all values in an image/data set
    by a set value. This function is typically used to increase the 
    distribution of the volume histogram. For example: a volume with peaks at 0, 
    10 and 25, multiplied by 5 would result in a volume with peaks at 0, 50, and
    125.
    This operation can be very useful in segmentation or phase seperation and 
    is expected to be used often in our image processing routines.

    Parameters
    ----------
    src_data : array
        Specifies the data set to be manipulated

    offset_data : {array, int, float}
        Specifies the data set or value withwhich to offset src_data_1

    Returns
    -------
    output : array
        Returns the resulting array to the designated variable

    Example
    -------
    result = mathops.multiply(img_1, img_2)
    """
    output = src_data * img_offset
    return output


def divide(src_data,
           offset_data):
    """
    This function enables the division of EITHER one image or volume data set by 
    another, OR division of all values in an image/data set by a set value. 
    This function is typically used for data set normalization purposes, or 
    basic  isolation of objects or materials/phases in a data set.

    Parameters
    ----------
    src_data : array
        Specifies the data set to be manipulated

    offset_data : {array, int, float}
        Specifies the data set or value withwhich to offset src_data_1

    Returns
    -------
    output : array
        Returns the resulting array to the designated variable

    Example
    -------
    result = mathops.divide(img_1, img_2)
    """
    if img_offset != 0:
        output = src_data / img_offset
    else:
        raise ValueError
    return output


def logical_and(src_data1,
                src_data2):
    """
    This function enables the computation of the LOGICAL_AND of two image or
    volume  data sets. This function can be used for data comparison, material
    isolation, noise removal, or mask application/generation.
    NOTE:
    The actual operator in the fuction can be the bitwise operator "&" as in
    "x & y = z". The function also works using the standard logical operator
    "and" as in "x and y = z".

    Parameters
    ----------
    src_data1 : array
        Specifies the first reference data

    src_data2 : array
        Specifies the second reference data

    Returns
    -------
    output : array
        Returns the resulting array to the designated variable

    Example
    -------
    result = mathops.logical_AND('img_1', 'img_2')
	"""
    output = np.logical_and(src_data1,
                            src_data2)
    return output


def logical_or(src_data1,
               src_data2):
    """
    This function enables the computation of the LOGICAL_OR of two image or
    volume data sets. This function can be used for data comparison,
    material isolation, noise removal, or mask application/generation.
    NOTE:
    The actual operator in the fuction can be the bitwise operator "|" as in
    "x | y = z". The function also works using the standard logical operator
    "or" as in "x or y = z".


    Parameters
    ----------
    src_data1 : array
        Specifies the first reference data

    src_data2 : array
        Specifies the second reference data

    Returns
    -------
    output : array
        Returns the resulting array to the designated variable

    Example
    -------
    result = mathops.logical_OR('img_1', 'img_2')
    """
    output = np.logical_or(src_data1,
                           src_data2)
    return output


def logical_not(src_data):
    """
    This function enables the computation of the LOGICAL_NOT of an image or
    volume data set. This function can be used for data comparison, material
    isolation, noise removal, or mask application/generation, and operates as a
    simple inversion of a binary data set.
    NOTE:
    The actual operator in the fuction can be the bitwise operator "~" as in
    "~x = z". The function also works using the standard logical operator "not"
    as in "not x = z"

    Parameters
    ----------
    src_data1 : array
        Specifies the reference data to be 'inverted'

    Returns
    -------
    output : array
        Returns the resulting array to the designated variable

    Example
    -------
    result = mathops.logical_NOT('img_1')
    """
    output = np.logical_not(src_data)
    return output


def logical_nand(src_data1,
                 src_data2):
    """
    This function enables the computation of the LOGICAL_NAND of two image or 
    volume data sets. This function enables easy isolation of all data points 
    NOT INCLUDED IN BOTH SOURCE DATA SETS. This function can be used for data 
    comparison, material isolation, noise removal, or mask 
    application/generation.

    Parameters
    ----------
    src_data1 : array
        Specifies the first reference data

    src_data2 : array
        Specifies the second reference data

    Returns
    -------
    output : array
        Returns the resulting array to the designated variable

    Example
    -------
    result = mathops.logical_NAND('img_1', 'img_2')
    """
    output = np.logical_not(np.logical_and(src_data1, 
                                           src_data2))
    return output


def logical_xor(src_data1,
                src_data2):
    """
    This function enables the computation of the LOGICAL_XOR of two image or
    volume data sets. This function will evaluate as true for all "true"
    voxels located in only one or the other data set. This function can be used
    for data comparison, material isolation, noise removal, or mask
    application/generation.
    NOTE:
    The actual operator in the fuction defenition can be "^" as in "x ^ y = z"

    Parameters
    ----------
    src_data1 : array
        Specifies the first reference data

    src_data2 : array
        Specifies the second reference data

    Returns
    -------
    output : array
        Returns the resulting array to the designated variable

    Example
    -------
    result = mathops.logical_XOR('img_1', 'img_2')
    """
    output = np.logical_xor(src_data1,
                            src_data2)
    return output


def logical_subtract(src_data1,
                     src_data2):
    """
    This function enables LOGICAL SUBTRACTION of one binary image or volume data 
    set from another. This function can be used to remove phase information, 
    interface boundaries, or noise, present in two data sets, without having to 
    worry about mislabeling of pixels which would result from arithmetic 
    subtraction. This function will evaluate as true for all "true" voxels 
    present ONLY in Source Dataset 1. This function can be used for data 
    cleanup, or boundary/interface analysis.

    Parameters
    ----------
    src_data1 : array
        Specifies the first reference data

    src_data2 : array
        Specifies the second reference data

    Returns
    -------
    output : array
        Returns the resulting array to the designated variable

    Example
    -------
    result = mathops.logical_SUB('img_1', 'img_2')
    """
    temp = np.logical_not(np.logical_and(src_data1, 
                                         src_data2))
    output = np.logical_and(src_data1, 
                            temp)
    return output


