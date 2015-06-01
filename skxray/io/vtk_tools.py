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
from IPython.display import Image


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

def np_to_vtk(input_array, pixel_spacing=None):
    """
    This function converts a given numpy array into a VTK object of the same
    type.

    Parameters
    ----------
    input_array : ndarray
    Complete path to the file to be loaded into memory

    pixel_spacing : tuple, optional
    Tuple containing pixel spacing, or resolution along the three primary
    axes (x, y, z). If pixel_spacing is not defined then values default to
    (1,1,1).

    Returns
    -------
    output : array-like
    """
    dataImporter = vtk.vtkImageImport()
    data_string = input_array.tostring()
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    _NP_TO_VTK_dTYPE_DICT = {
        'bool' : dataImporter.SetDataScalarTypeToUnsignedChar(),
        'character' : dataImporter.SetDataScalarTypeToUnsignedChar(),
        'uint8' : dataImporter.SetDataScalarTypeToUnsignedChar(),
        'uint16' : dataImporter.SetDataScalarTypeToUnsignedShort(),
        'uint32' : dataImporter.SetDataScalarTypeToInt(),
        'uint64' : dataImporter.SetDataScalarTypeToInt(),
        'int8' : dataImporter.SetDataScalarTypeToShort(),
        'int16' : dataImporter.SetDataScalarTypeToShort(),
        'int32' : dataImporter.SetDataScalarTypeToInt(),
        'int64' : dataImporter.SetDataScalarTypeToInt(),
        'float32' : dataImporter.SetDataScalarTypeToFloat(),
        'float64' : dataImporter.SetDataScalarTypeToDouble(),
        }
    input_array_shape = input_array.shape
    _NP_TO_VTK_dTYPE_DICT[str(input_array.dtype)]
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(1)
    dataImporter.SetDataExtent(0, input_array_shape[2],
                               0, input_array_shape[1],
                               0, input_array_shape[0])
    dataImporter.SetWholeExtent(0, input_array_shape[2],
                                0, input_array_shape[1],
                                0, input_array_shape[0])
    if pixel_spacing == None:
        pixel_spacing = [1, 1, 1]
    dataImporter.SetDataSpacing(pixel_spacing[0], pixel_spacing[1],
                                pixel_spacing[2])
    return dataImporter


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

def ipython_vtk_viewer(renderer, width=None, height=None):
    """
    Converts a vtkRenderer object into an iPython image

    Note: This method generates an external window within which
    rendering actually occurs. The generated image is then returned
    to the ipython session.

    Ref: https://pyscience.wordpress.com/2014/09/03/ipython-notebook-vtk/

    Parameters
    ----------
    renderer :

    width : int, optional

    height : int, optional

    Returns
    -------

    """
    if width == None:
        width = 400
    if height == None:
        height=300

    #Create new rendering window
    renderWindow = vtkRenderWindow()
    #Enable off-screen rendering
    renderWindow.SetOffScreenRendering(1)
    #Create instance
    renderWindow.AddRenderer(renderer)
    #Set Dims
    renderWindow.SetSize(width, height)
    #Execute render operation
    renderWindow.Render()

    #Create a new vtkWindow -> Image filter object which allows us to read
    # the data in a vtkWindow and use it as input to the imaging pipeline.
    windowToImageFilter = vtk.vtkWindowToImageFilter()
    #Add renderWindow to the filter object
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.Update() #Update object

    #Create vtk PNG writer object thereby allowing creation of PNG images
    # of the vtkWindow rendering
    writer = vtk.vtkPNGWriter()
    #Setting eq to 1 enables image to be held in memory instead of saving
    # to disk. This is the desired option since the image is meant to be
    # loaded back into the notebook instead of stored on disk
    writer.SetWriteToMemory(1)
    # Link between output (GetOutputPort) of vtkWindowToImageFilter and
    # the PNG writer (SetInputConnection)
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()
    data = str(buffer(writer.GetResult()))

    return Image(data)


# A simple function to be called when the user decides to quit the application.
def exitCheck(obj, event):
    if obj.GetEventPending() != 0:
        obj.SetAbortRender(1)


def vtk_vis_props(dataImporter):
    """
    Parameters
    ----------
    dataImporter : array

    Returns
    -------
    volumeProperty : array
    """
    dataImporter.SetDataExtent(0, 74, 0, 74, 0, 74)
    dataImporter.SetWholeExtent(0, 74, 0, 74, 0, 74)
    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    alphaChannelFunc.AddPoint(0, 0.0)
    alphaChannelFunc.AddPoint(50, 0.05)
    alphaChannelFunc.AddPoint(100, 0.1)
    alphaChannelFunc.AddPoint(150, 0.2)

    colorFunc = vtk.vtkColorTransferFunction()
    colorFunc.AddRGBPoint(50, 1.0, 0.0, 0.0)
    colorFunc.AddRGBPoint(100, 0.0, 1.0, 0.0)
    colorFunc.AddRGBPoint(150, 0.0, 0.0, 1.0)

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorFunc)
    volumeProperty.SetScalarOpacity(alphaChannelFunc)
    return volumeProperty


def vtk_viewer(volume_data, volumeProperty, width=None, height=None):
    """
    Create a VTK viewer window to display 3D rendered surface and volume
    objects.

    Parameters
    ----------
    volume_data : array

    volumeProperty : array

    width : int, optional

    height : int, optional

    Returns
    -------

    """
    if width == None:
        width = 400
    if height == None:
        height=300

    # This class describes how the volume is rendered (through ray tracing).
    compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
    # We can finally create our volume. We also have to specify the data for
    # it, as well as how the data will be rendered.
    volumeMapper = vtk.vtkVolumeRayCastMapper()
    volumeMapper.SetVolumeRayCastFunction(compositeFunction)
    volumeMapper.SetInputConnection(volume_data.GetOutputPort())

    # The class vtkVolume is used to pair the preaviusly declared volume as
    # well as the properties to be used when rendering that volume.
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    #Create renderer instance
    ren = vtk.vtkRenderer()
    # add the volume to the renderer
    ren.AddVolume(volume)
    # set background color
    ren.SetBackground(0.5,0.5,0.5)

    #Create new rendering window
    renWin = vtk.vtkRenderWindow()
    #Create instance
    renWin.AddRenderer(ren)
    #Set Dims
    renWin.SetSize(width, height)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    renWin.AddObserver("AbortCheckEvent", exitCheck)
    iren.Initialize()
    renWin.Render()
    iren.Start()


def write_stl(vtk_obj, filename, path=None):
    """
    This function enables the writing, or exporting of surface renderings as
    stl files. These files can be saved for further analysis using VTK,
    MayaVI, Paraview, or any 3D graphics software that is able to read this
    common, and standard surface file type.

    Parameters
    ----------
    surf_obj : array-like

    filename : str

    path : str, optional

    Returns
    -------

    """
    #Confirm that object is a VTK surface object, and if not, then attempt
    # conversion.

    #Generate writer object
    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(vtk_obj.GetOutputPort())
    writer.SetFileTypeToBinary()
    if path != None:
        filename = path+filename
    writer.SetFileName(filename)
    writer.Write()

