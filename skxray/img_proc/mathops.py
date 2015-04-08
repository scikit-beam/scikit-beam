# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Oct. 2013
"""
# This class is designed to facilitate image arithmetic and logical operations
# on image data sets.
"""

import numpy as np


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


