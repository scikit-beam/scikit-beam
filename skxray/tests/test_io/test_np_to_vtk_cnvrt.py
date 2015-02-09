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
import vtk
from skxray.io import np_to_vtk_cnvrt as np2vtk
from numpy.testing import assert_equal

_NP_TO_VTK_dTYPE_DICT = {
    #np.bool: vtk.VTK_BIT,
    #np.character: vtk.VTK_UNSIGNED_CHAR,
    np.uint8: vtk.VTK_UNSIGNED_CHAR,
    np.uint16: vtk.VTK_UNSIGNED_SHORT,
    np.uint32: vtk.VTK_UNSIGNED_INT,
    np.uint64: vtk.VTK_UNSIGNED_LONG,
    np.int8: vtk.VTK_CHAR,
    np.int16: vtk.VTK_SHORT,
    np.int32: vtk.VTK_INT,
    np.int64: vtk.VTK_LONG,
    np.float32: vtk.VTK_FLOAT,
    np.float64: vtk.VTK_DOUBLE,
    #np.complex64: vtk.VTK_FLOAT,
    #np.complex128: vtk.VTK_DOUBLE
}


VTK_ID_TYPE_SIZE = vtk.vtkIdTypeArray().GetDataTypeSize()
if VTK_ID_TYPE_SIZE == 4:
    ID_TYPE_CODE = np.int32
elif VTK_ID_TYPE_SIZE == 8:
    ID_TYPE_CODE = np.int64

VTK_LONG_TYPE_SIZE = vtk.vtkLongArray().GetDataTypeSize()
if VTK_LONG_TYPE_SIZE == 4:
    LONG_TYPE_CODE = np.int32
    ULONG_TYPE_CODE = np.uint32
elif VTK_LONG_TYPE_SIZE == 8:
    LONG_TYPE_CODE = np.int64
    ULONG_TYPE_CODE = np.uint64


_VTK_TO_NP_dTYPE_DICT = {
    #vtk.VTK_BIT : np.bool,
    vtk.VTK_CHAR : np.int8,
    vtk.VTK_UNSIGNED_CHAR : np.uint8,
    vtk.VTK_SHORT : np.int16,
    vtk.VTK_UNSIGNED_SHORT : np.uint16,
    vtk.VTK_INT : np.int32,
    vtk.VTK_UNSIGNED_INT : np.uint32,
    vtk.VTK_LONG : LONG_TYPE_CODE,
    vtk.VTK_LONG_LONG : np.int64,
    vtk.VTK_UNSIGNED_LONG : ULONG_TYPE_CODE,
    vtk.VTK_UNSIGNED_LONG_LONG : np.uint64,
    vtk.VTK_ID_TYPE : ID_TYPE_CODE,
    vtk.VTK_FLOAT : np.float32,
    vtk.VTK_DOUBLE : np.float64
}

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


def test_vtk_conversion():
    """
    This function tests and verifies conversion of synthetic 2D and
    3D numpy arrays to the appropriate and corresponding VTK data object.
    The current list of numpy data types included, along with the
    corresponding VTK data type is as follows:

        Numpy dType  :  VTK dType
        -----------     ---------
        bool         :  VTK_BIT
        character    :  VTK_UNSIGNED_CHAR
        uint8        :  VTK_UNSIGNED_CHAR
        uint16       :  VTK_UNSIGNED_SHORT
        uint32       :  VTK_UNSIGNED_INT
        uint64       :  VTK_UNSIGNED_LONG_LONG
        int8         :  VTK_CHAR
        int16        :  VTK_SHORT
        int32        :  VTK_INT
        int64        :  VTK_LONG_LONG
        float32      :  VTK_FLOAT
        float64      :  VTK_DOUBLE
        complex64    :  VTK_FLOAT
        complex128   :  VTK_DOUBLE

    Special Note for uint64 and int64 type conversion:
        Type conversion to VTK_UNSIGNED_LONG_LONG and VTK_LONG_LONG
        generates an error. However, conversion to VTK_LONG and
        VTK_UNSIGNED_LONG appears to work.
    """

    for _ in _NP_TO_VTK_dTYPE_DICT.keys():
        test_np_array = np.array(10*np.random.rand(3,3,3), dtype=_)
        array_shape = test_np_array.shape
        vtk_obj = np2vtk.ndarray_to_vtk_obj(test_np_array,
                                            array_type=_NP_TO_VTK_dTYPE_DICT[_])
        assert_equal(_VTK_DTYPE_INDEX_DICT[vtk_obj.GetDataType()],
                     _NP_TO_VTK_dTYPE_DICT[_])
        np_result = np2vtk.vtk_to_np(vtk_obj, array_shape)
        print np_result.dtype
        print np_result.shape
        assert_equal(np_result, test_np_array)



def test_ndarray_to_vtk_obj():
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
    for _ in _VTK_TO_NP_dTYPE_DICT.keys():


