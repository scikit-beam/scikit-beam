# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, March 2015
"""
This module contains test functions for the numpy-to-vtk, and vtk-to-numpy
functions that are used in order to render and visualize 3D data in
Vistrails. These functions are required due to the design decision to carry
data through our pipelines using Numpy Arrays.
"""
import numpy as np
import vtk
from skxray.io import vtk_tools
from numpy.testing import assert_equal

dataImporter = vtk.vtkImageImport()

_NP_TO_VTK_dTYPE_DICT = {
    'bool': 3,  # vtk.VTK_UNSIGNED_CHAR
    #'str': 2,  # vtk.VTK_CHAR
    'uint8': 3,  # vtk.VTK_UNSIGNED_CHAR
    'uint16': 5,  # vtk.VTK_UNSIGNED_SHORT
    'uint32': 7,  # vtk.VTK_UNSIGNED_INT
    'uint64': 17,  # vtk.VTK_UNSIGNED_LONG_LONG
    'int8': 15,  # vtk.VTK_SIGNED_CHAR
    'int16': 4,  # vtk.VTK_SHORT
    'int32': 6,  # vtk.VTK_INT,
    'int64': 16,  # vtk.VTK_LONG_LONG,
    'float32': 10,  # vtk.VTK_FLOAT
    'float64': 11,  # vtk.VTK_DOUBLE
}

VTK_ID_TYPE_SIZE = vtk.vtkIdTypeArray().GetDataTypeSize()
if VTK_ID_TYPE_SIZE == 4:
    ID_TYPE_CODE = 'int32'
elif VTK_ID_TYPE_SIZE == 8:
    ID_TYPE_CODE = 'int64'

VTK_LONG_TYPE_SIZE = vtk.vtkLongArray().GetDataTypeSize()
if VTK_LONG_TYPE_SIZE == 4:
    LONG_TYPE_CODE = 'int32'
    ULONG_TYPE_CODE = 'uint32'
elif VTK_LONG_TYPE_SIZE == 8:
    LONG_TYPE_CODE = 'int64'
    ULONG_TYPE_CODE = 'uint64'

_VTK_TO_NP_dTYPE_DICT = {
        #vtk.VTK_CHAR: 'str',
        vtk.VTK_UNSIGNED_CHAR: 'uint8',
        vtk.VTK_UNSIGNED_SHORT: 'uint16',
        vtk.VTK_UNSIGNED_INT: 'uint32',
        vtk.VTK_UNSIGNED_LONG_LONG: 'uint64',
        vtk.VTK_SIGNED_CHAR: 'int8',
        vtk.VTK_SHORT: 'int16',
        vtk.VTK_INT: 'int32',
        vtk.VTK_LONG_LONG: 'int64',
        vtk.VTK_FLOAT: 'float32',
        vtk.VTK_DOUBLE: 'float64',
        }

_VTK_DTYPE_INDEX_DICT = {
    0: vtk.VTK_VOID,
    1: vtk.VTK_BIT,
    2: vtk.VTK_CHAR,
    15: vtk.VTK_SIGNED_CHAR,
    3: vtk.VTK_UNSIGNED_CHAR,
    4: vtk.VTK_SHORT,
    5: vtk.VTK_UNSIGNED_SHORT,
    6: vtk.VTK_INT,
    7: vtk.VTK_UNSIGNED_INT,
    8: vtk.VTK_LONG,
    9: vtk.VTK_UNSIGNED_LONG,
    10: vtk.VTK_FLOAT,
    11: vtk.VTK_DOUBLE,
    12: vtk.VTK_ID_TYPE,
    13: vtk.VTK_STRING,
    14: vtk.VTK_OPAQUE,
    16: vtk.VTK_LONG_LONG,
    17: vtk.VTK_UNSIGNED_LONG_LONG,
    18: vtk.VTK___INT64,
    19: vtk.VTK_UNSIGNED___INT64,
    20: vtk.VTK_VARIANT,
    21: vtk.VTK_OBJECT,
    22: vtk.VTK_UNICODE_STRING
}


def test_np_to_vtk_convert():
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
        test_np_array = np.array(10 * np.random.rand(3, 3, 3), dtype=_)
        array_shape = test_np_array.shape
        vtk_obj = vtk_tools.np_to_vtk(test_np_array)
        if str(test_np_array.dtype) == 'bool':
            conversion_dtype = 'uint8'
        else:
            conversion_dtype = str(test_np_array.dtype)
        assert_equal(_VTK_TO_NP_dTYPE_DICT[vtk_obj.GetDataScalarType()],
                     conversion_dtype)
        #TODO:
        #np_result = vtk_tools.vtk_to_np(vtk_obj, array_shape)
        #print np_result.dtype
        #print np_result.shape
        #assert_equal(np_result, test_np_array)


def test_vtk_to_ndarray():
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
    #for _ in _VTK_TO_NP_dTYPE_DICT.keys():
