# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Nov. 2013
"""
This module contains all fileIO operations and file conversion for the image 
processing tool kit in pyLight. Operations for loading, saving, and converting 
files and data sets are included for the following file formats:

2D and 3D TIFF (.tif and .tiff)
RAW (.raw and .volume)
HDF5 (.h5 and .hdf5)
"""

import numpy as np
from vtk.util import numpy_support
import vtk


_NP_TO_VTK_dTYPE_DICT = {
    'bool' : vtk.VTK_BIT,
    'character' : vtk.VTK_UNSIGNED_CHAR,
    'uint8' : vtk.VTK_UNSIGNED_CHAR,
    'uint16' : vtk.VTK_UNSIGNED_SHORT,
    'uint32' : vtk.VTK_UNSIGNED_INT,
    'uint64' : vtk.VTK_UNSIGNED_LONG_LONG,
    'int8' : vtk.VTK_CHAR,
    'int16' : vtk.VTK_SHORT,
    'int32' : vtk.VTK_INT,
    'int64' : vtk.VTK_LONG_LONG,
    'float32' : vtk.VTK_FLOAT,
    'float64' : vtk.VTK_DOUBLE,
    }
#
#
# _VTK_TO_NP_dTYPE_DICT = {
#     vtk.VTK_BIT:numpy.bool,
#     vtk.VTK_CHAR:numpy.int8,
#     vtk.VTK_UNSIGNED_CHAR:numpy.uint8,
#     vtk.VTK_SHORT:numpy.int16,
#     vtk.VTK_UNSIGNED_SHORT:numpy.uint16,
#     vtk.VTK_INT:numpy.int32,
#     vtk.VTK_UNSIGNED_INT:numpy.uint32,
#     vtk.VTK_LONG:LONG_TYPE_CODE,
#     vtk.VTK_LONG_LONG:numpy.int64,
#     vtk.VTK_UNSIGNED_LONG:ULONG_TYPE_CODE,
#     vtk.VTK_UNSIGNED_LONG_LONG:numpy.uint64,
#     vtk.VTK_ID_TYPE:ID_TYPE_CODE,
#     vtk.VTK_FLOAT:numpy.float32,
#     vtk.VTK_DOUBLE:numpy.float64
# }


_VTK_DTYPE_INDEX_DICT = {
    0 : vtk.VTK_VOID,
    1 : vtk.VTK_BIT,
    2 : vtk.VTK_CHAR,
    15 : vtk.VTK_SIGNED_CHAR,
    3 : vtk.VTK_UNSIGNED_CHAR,
    4 : vtk.VTK_SHORT,
    5 : vtk.VTK_UNSIGNED_SHORT,
    6 : vtk.VTK_INT,
    7 : vtk.VTK_UNSIGNED_INT,
    8 : vtk.VTK_LONG,
    9 : vtk.VTK_UNSIGNED_LONG,
    10 : vtk.VTK_FLOAT,
    11 : vtk.VTK_DOUBLE,
    12 : vtk.VTK_ID_TYPE,
    13 : vtk.VTK_STRING,
    14 : vtk.VTK_OPAQUE,
    16 : vtk.VTK_LONG_LONG,
    17 : vtk.VTK_UNSIGNED_LONG_LONG,
    18 : vtk.VTK___INT64,
    19 : vtk.VTK_UNSIGNED___INT64,
    20 : vtk.VTK_VARIANT,
    21 : vtk.VTK_OBJECT,
    22 : vtk.VTK_UNICODE_STRING
    }

def np_to_vtk(src_data):
    """
    This function converts a given numpy array into a VTK object of the same
    type.

    Parameters
    ----------
    src_data : ndarray
	Complete path to the file to be loaded into memory

    Returns
    -------
    output : array-like
    """
    src_data_shape = src_data.shape
    print src_data.dtype
    if src_data.dtype == bool:
        src_data = np.array(src_data, dtype='uint8')
    vtk_obj = numpy_support.numpy_to_vtk(num_array=src_data.ravel(),
                                          deep=True)
    return vtk_obj


def vtk_to_np(src_data, shape=None):
    """
    This function automates the saving of volumes as .tiff files using a single
    keyword

    Parameters
    ----------
    file_name : str
        Specify the path and file name to which you want the volume saved

    data : array
        Specify the array to be saved

    Returns
    -------

    """
    np_obj = numpy_support.vtk_to_numpy(src_data)
    if shape != None:
        np_obj = np.reshape(np_obj, shape)
    return np_obj

