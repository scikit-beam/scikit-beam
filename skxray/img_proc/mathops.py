# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Oct. 2013
"""
This module is designed to facilitate image arithmetic and logical operations
on image data sets. The included functions supplement the logical operations
currently provided in numpy in order to provide a complete set of logical
operations for data analysis.

The new functions include:
    logical_nand: Identifies all elements NOT included in BOTH inputs
    logical_nor: Identifies all elements NOT included in EITHER input
    logical_sub: Identifies all elements ONLY included in input_1
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from numpy import (logical_and, logical_or, logical_not, logical_xor, add,
                subtract, multiply, divide)


def logical_nand(x1,
                 x2):
    """
    This function enables the computation of the LOGICAL_NAND of two image or
    volume data sets. This function enables easy isolation of all data points
    NOT INCLUDED IN BOTH SOURCE DATA SETS. This function can be used for data
    comparison, material isolation, noise removal, or mask
    application/generation.

    Parameters
    ----------
    x1 : array_like
        Specifies the first reference data

    x2 : array_like
        Specifies the second reference data

    Returns
    -------
    output : {ndarray, bool}
        Returns the resulting array to the designated variable

    Example
    -------
    >>>input_1 = [[0,0,1,0,0], [2,1,1,1,2], [2,0,1,0,2]]
    >>>input_2 = [[0,0,0,0,0], [2,1,1,1,2], [0,0,0,0,0]]
    result = logical_nand(input_1, input_2)
    >>>result
    [out]: array([[ True,  True,  True,  True,  True],
                  [False, False, False, False, False],
                  [ True,  True,  True,  True,  True]], dtype=bool)
    """
    output = logical_not(logical_and(x1, x2))
    return output


def logical_nor(x1,
                x2):
    """
    This function enables the computation of the LOGICAL_NOR of two image or
    volume data sets. This function enables easy isolation of all data points
    NOT INCLUDED IN EITHER OF THE SOURCE DATA SETS. This function can be used
    for data comparison, material isolation, noise removal, or mask
    application/generation.

    Parameters
    ----------
    x1 : array_like
        Specifies the first reference data

    x2 : array_like
        Specifies the second reference data

    Returns
    -------
    output : {ndarray, bool}
        Returns the resulting array to the designated variable

    Example
    -------
    >>>input_1 = [[0,0,1,0,0], [2,1,1,1,2], [2,0,1,0,2]]
    >>>input_2 = [[0,0,0,0,0], [2,1,1,1,2], [0,0,0,0,0]]
    result = logical_not(input_1, input_2)
    >>>result
    [out]: array([[ True,  True, False,  True,  True],
                  [False, False, False, False, False],
                  [False,  True, False,  True, False]], dtype=bool)
    """
    output = logical_not(logical_or(x1, x2))
    return output


def logical_sub(x1,
                x2):
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
    x1 : array_like
        Specifies the first reference data

    x2 : array_like
        Specifies the second reference data

    Returns
    -------
    output : {ndarray, bool}
        Returns the resulting array to the designated variable

    Example
    -------
    >>>input_1 = [[0,0,1,0,0], [2,1,1,1,2], [2,0,1,0,2]]
    >>>input_2 = [[0,0,0,0,0], [2,1,1,1,2], [0,0,0,0,0]]
    result = logical_nand(input_1, input_2)
    >>>result
    [out]: array([[False, False,  True, False, False],
                  [False, False, False, False, False],
                  [ True, False,  True, False,  True]], dtype=bool)

    """
    temp = logical_not(logical_and(x1, x2))
    output = logical_and(x1,
                         temp)
    return output


__all__ = ["add", "subtract", "multiply", "divide", "logical_and",
           "logical_or", "logical_nor", "logical_xor", "logical_not",
           "logical_sub", "logical_nand"]
