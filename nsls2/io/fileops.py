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
"""
REVISION LOG: (FORMAT: "PROGRAMMER INITIALS: DATE -- RECORD")
-------------------------------------------------------------
 gci: 11/26/2013 -- Conversion of the original module iops.py to a class-based
    module for direct implementation in the PyLight software package. Class
    conversion is the second step in adding/implementing new functions in the
    web-based package.
 gci: 2/7/14 -- Updating file for pull request to GITHUB.
    1) Added new function: createIP_HDF5
    This function enables creating image processing specific HDF5 files
    based off of the outline specification documentation provided in:
    'The Scientific Data Exchange Introductory Guide'
    http://www.aps.anl.gov/DataExchange
    2) Added new function: convert_CT_2_HDF5
    This function enables the conversion of other image and volume file
    formats into our prescribed HDF5 file format
    3) Added new function: open_H5_file
    This is a basic function that opens HDF5 files that have already
    been created in our prescribed format and defines that basic groups
    and data sets that will already have been included in the file.
    NOTE: THIS FUNCTION MAY SIMPLY BE REDUNDANT AND UNNECESSARY. I ADDED
    IT THINKING THAT IT COULD BE TAILORED TO SIMPLIFY ADDITIONAL CODING.
    4) Added new function: close_H5_file
    This is a basic function that simplifies the closing of our HDF5 files
    NOTE: THIS FUNCTION MAY SIMPLY BE REDUNDANT AND UNNECESSARY. I ADDED
    IT THINKING THAT IT COULD BE TAILORED TO SIMPLIFY ADDITIONAL CODING.
    5) Altered the structure of the module so that functions are no longer
    included in a class, but are stand alone functions. I may need to
    switch back to a class structure, but am unsure at the moment.
    6) Changed the order of input variables for load_RAW from (z,y,x,file_name)
    to (file_name, z, y, x)
    7) Added complete descriptions of each of the functions included
    in this module.
GCI: 2/12/14 -- re-inserted function for saving volumes as .tiff files
    (save_data_Tiff). This function was included in the original iops.py
    file, but never made it into the C1_fileops module.
GCI: 2/18/14 -- 1) Converted documentation to docstring format and changed file
    name to fileops.py.
    2) Removed calls to other modules from within each function and placed in
    a separate group which loads any time this module is imported.
GCI: 3/11/14 -- 1) Added h5_obj_search() function which searches the specified
    H5 group and returns the next, unused, object name in the group, from which
    the next object produced through the image processing workflow can be
    written or saved.
    2) Added h5_fName_dict() which defines the group structure of an IP_H5 file
    as a searchable dictionary and contains the association between the data
    type group and the associated base file name for data sets.
GCI: 5/13/14 -- Added load function for netCDF files. Specifically for loading
    data sets acquired at the APS Sector 13 beamline.
"""

import numpy as np
import tifffile
import h5py
from netCDF4 import Dataset


def load_RAW(file_name,
             z,
             y,
             x):
    """
    This function loads the specified RAW file format data set (.raw, or
    .volume extension) file into a numpy array for further analysis.

    Parameters
    ----------
    file_name: string
    Complete path to the file to be loaded into memory

    z: integer
    Z-axis array dimension as an integer value

    y: integer
    Y-axis array dimension as an integer value

    x: integer
    X-axis array dimension as an integer value

    Returns
    -------
    output: NxN or NxNxN ndarray
    Returns the loaded data set as a 32-bit float numpy array

    Example
    -------
    vol = C1_fileops.load_RAW('file_path', 520, 695, 695)
    """
    src_volume = np.empty((z,
                           y,
                           x), np.float32)
    print('loading ' + file_name)
    src_volume.data[:] = open(file_name).read()  # sample file is now loaded
    target_var = src_volume[:, :, :]
    print 'Volume loaded successfully'
    return target_var


def load_netCDF(file_name, data_type=None):
    # TODO: Write this as a class so that you can either retreive just the
    # volume or just the dictionary. This should make iterable volume loading
    # more straight forward and easy.
    src_file = Dataset(file_name)
    data = src_file.variables['VOLUME']
    tmp_dict = src_file.__dict__
    try:
        print 'Data acquisition scale factor: ' + str(data.scale_factor)
        scale_value = data.scale_factor
        print 'Image data values adjusted by the recorded scale factor.'
    except:
        print 'the scale factor is non existent'
        scale_value = 1.0
    data = data / scale_value
    return data, tmp_dict


def load_tif(file_name):
    """
    This function loads a tiff file into memory as a numpy array of the same
    type as defined in the tiff file. This function is able to load both 2-D
    and 3-D tiff files. File extensions .tif and .tiff both work in the
    execution of this function.
    NOTE:
    Requires additional module -- tifffile.py
    Initial attempts at loading tiff files forcused on using the imageio
    module. However, this module was found to be limited in scope to
    2-dimensional tif files/arrays. Additional searching came up with a
    different module identified as tifffile.py. This module has been developed
    by the Fluorescence Spectroscopy group at UC-Irvine. After installing this
    module, a simple call to tifffile.imread(file_path) successfully loads
    both 2-D and 3-D tiff files, and the module appears to be able to identify
    and load files with both tiff extensions (tif and tiff). As of 10/29/13,
    I've changed the loading code to focus solely on implementation of the
    tifffile module.

    Parameters
    ----------
     file_name: string
    Complete path to the file to be loaded into memory

    Returns
    -------
    output: NxN or NxNxN ndarray
    Returns a numpy array of the same data type as the original tiff file

    Example
    -------
    vol = fileops.load_tif('file_path')

    """
    print('loading ' + file_name)
    target_var = tifffile.imread(file_name)
    # print imageio.volread(file_name)
    print 'Volume loaded successfully'
    return target_var


def save_data_Tiff(file_name,
                   data):
    """
    This function automates the saving of volumes as .tiff files using a single
    keyword

    Parameters
    ----------
    file_name: Specify the path and file name to which you want the volume
        saved

    data: numpy array
        Specify the array to be saved

    Returns
    -------
    image file: 2D or 3D tiff image or image stack
        Saves the specified volume as a 3D tif (or if the data is a 2D image
        then data saved as 2D tiff).
    """
    tifffile.imsave(file_name,
                    data)
    print "File Saved as: " + file_name


# Code to convert common CT Volume data formats to HDF5 Standard
def createIP_HDF5(base_file_name):
    """
    This function creates an HDF5 file using the prescribed format for our
    image processing specific HDF5 files, complete with the baseline groups:
    1) "exchange"
    2) "measurements"
    3) "provenance"
    4) "implements"
    The newly created HDF5 file is then returned for data inclusion or further
    modification.

    Parameters
    ----------
    base_file_name: string
    Specify the name for the file to be created
    NOTE:
    Do not include the .h5 or .hdf5 extension as this will be added during
    file creation.

    Returns
    -------
    output: Open h5py.File
    Function returns the open and newly created HDF5 file
    """
    f = h5py.File(base_file_name + '.h5', 'w')
    grp1 = f.create_group("exchange")
    grp2 = f.create_group("measurements")
    grp3 = f.create_group("provenance")
    grp4 = f.create_group("workflow_cache")
    grp5 = f.create_group("products")
    imp = f.create_dataset("implements", data='exchange: measurements: workflow_cache: products: provenance')
    return f


def open_H5_file(H5_file):
    """
    This funciton opens a previously existing HDF5 file in read/write mode


    Parameters
    ----------
    H5_file: string
    Path to the target HDF5 file

    Returns
    -------
    output: Open h5py.File
    Returns the opened HDF5 file to a specified variable name

    Example
    -------
    f = open_H5_file('file_path')
    """
    f = h5py.File(H5_file, 'r+')
    return f


def close_H5_file(H5_file):
    """
    This function is a simple script to close an open HDF5 file using a single
    keyword.


    Parameters
    ----------
    H5_file: h5py.File
    Variable name of the open HDF5 file
    """
    f = H5_file
    f.close()

def load_reconData(file_name,
                   x_dim=None,
                   y_dim=None,
                   z_dim=None,
           dim_assign="Auto"):
    """
    This function is intended to be a single, callable, function capable of
    loading any and all of the common file types used to store x-ray imaging
    data. Once executed, the function will auto-detect file type based on the
    file suffix and load the file using the appropriate loading function.
    List of current file extensions available to load using this function:
    .tif
    .tiff
    .raw
    .volume
    .h5
    .hdf5

    NOTE:
    If the target data set is of file typ RAW then axial dimensions must be
    added manually, or have been previously specified.
    An additional optional keyword (dim_assign) has been added in order to
    specify whether dimensions are to be assigned manualy or not. This setting
    should be assigned a value of "Auto" if dimensions are EITHER assigned when
    the function is called, OR are not required in order to load the file, as
    is the case for all file formats other than RAW.
    If the target file is of type RAW and dim_assign is set to "Manual" then
    the loading function is setup to request manual, command-line input for
    the X, Y, and Z axial dimensions of the volume or image data set.

    Parameters
    ----------
    file_name: string
    Complete path to the file to be loaded into memory

    x_dim: integer (OPTIONAL)
    X-axis data set dimension, required ONLY if data set is of type RAW
    AND dim_assign is set to "Auto". If data set is of type RAW and
    dim_assign is set to "Manual" then this value will need to be entered
    at the command line.

    y_dim: integer (OPTIONAL)
    Y-axis data set dimension, required ONLY if data set is of type RAW
    AND dim_assign is set to "Auto". If data set is of type RAW and
    dim_assign is set to "Manual" then this value will need to be entered
    at the command line.

    z_dim: integer (OPTIONAL)
    Z-axis data set dimension, required ONLY if data set is of type RAW
    AND dim_assign is set to "Auto". If data set is of type RAW and
    dim_assign is set to "Manual" then this value will need to be entered
    at the command line.

    dim_assign: string (OPTIONAL)
    Option: "Auto" -- Default
        This keyword is automatically set to "Auto" specifying that if
        volume dimensions are required in order to load a file
        (currently only applies to files of type RAW) then they will be
        entered in the function call string.
    Option: "Manual"
        If dim_assing is set to "Manual" then the axial dimensions for
        the volume to be loaded will be entered at command line prompts.
        This is currently only needed as an option for loading files of
        type RAW.

    Returns
    -------
    output: NxN, NxNxN numpy array, or HDF5 file
    Returns a numpy array or opened HDF5 file containing the source
    volume or image.

    Example
    -------
    vol = C1_fileops.load_reconData('file_path')
    """
    # TODO INCORPORATE os.path.splitext(path) and trim code-- will split path
    # so that file extensions are identified but split, count, search code is
    # unnecessary.
    fType_options = ['raw', 'volume', 'tif', 'tiff', 'am', 'h5', 'hdf5']
    src_fNameSep = file_name.split('.')
    dot_cnt = file_name.count('.')
    src_fType = src_fNameSep[dot_cnt]
    # TODO incorporate if not in "extension list" the raise
    # ValueException("blah blah")
    load_fType = [x for x in fType_options if x in src_fType]
    print "Loading ." + load_fType[0] + " file: " + file_name
    if dim_assign == "Manual":
        if load_fType[0] in ('raw', 'volume'):
            x_dim = input('Enter x-dimension:')
            y_dim = input('Enter y-dimension:')
            z_dim = input('Enter z-dimension:')
            target_var = load_RAW(z_dim, y_dim, x_dim, file_name)
    if dim_assign == "Auto":
        if load_fType[0] in ('raw', 'volume'):
            target_var = load_RAW(z_dim, y_dim, x_dim, file_name)
    elif load_fType[0] in ('tif', 'tiff'):
        target_var = load_tif(file_name)
    elif load_fType[0] in ('h5', 'hdf5'):
        target_var = open_H5_file(file_name)
    else:
        raise ValueException ("File type not recognized.")
    print "File loaded successfully"
    return target_var


def convert_CT_2_HDF5(H5_file,
                      src_data,
                      resolution,
                      units):
    """
    This function is used in the conversion of data in other file formats to
    our prescribed HDF5 file format.


    Parameters
    ----------
    H5_file: string
    The name of an empty HDF5 file (or possibly a new GROUP within an HDF5
    file, though this needs to be tested).

    src_data: string
    The name of the array containing the image data to be added to the HDF5
    file. This data will have been previously loaded using a function such
    as load_reconData().

    resolution: float
    Specify the linear pixel or voxel resolution for the data set.

    units: string
    Identify the units for the pixel or voxel resolution as a string.
    """
    f = H5_file
    grp1 = f["exchange"]
    grp2 = f["measurements"]
    grp3 = f["provenance"]
    imp = f["implements"]
    data1 = grp1.create_dataset("recon_data", data=src_data,
                                compression='gzip')
    z_dim, y_dim, x_dim = src_data.shape
    data1.attrs.create("x-dim", x_dim)
    data1.attrs.create("y-dim", y_dim)
    data1.attrs.create("z-dim", z_dim)
    data1.attrs.create("Resolution", resolution)
    data1.attrs.create("Units", units)
    f['exchange']['recon_data'].dims[0].label = 'z'
    f['exchange']['recon_data'].dims[1].label = 'y'
    f['exchange']['recon_data'].dims[2].label = 'x'
    f['exchange']['voxel_size'] = [resolution, resolution, resolution]
    f['exchange']['voxel_size'].attrs.create("Units", units)
    f['exchange']['recon_data'].dims.create_scale(f['exchange']['voxel_size'],
                                                  'Z, Y, X')
    return f


def h5_obj_search(h5_grp_path,
                   test_name_base):
    """
    This function searches through the identified HDF5 file group object and
    counts the number of objects (typically, data sets) that have already been
    created in the specified HDF5 group. This function is currently used to
    write image processing operation results as a new data set without having
    executables or the user need to explicitly specify the new object name.


    Parameters
    ----------
    h5_grp_path: string
        The path, internal to the target HDF5 file, that needs to be seached.

    test_name_base: string
        The base object name to be counted, sorted or referenced.

    Returns
    -------
    counter: integer
        The number of objects with the specified base name currently contained
        in the group AND the index number for the next object name in an
        iterable series, assuming that the starting index number is 0 (zero).

    next_obj_name: string
        The output is the next iterable object name that has not yet been
        created
    """
    counter = 0
    for x in h5_grp_path.keys():
        if test_name_base in h5_grp_path.keys()[counter]:
            counter += 1
        else:
            continue
    next_obj_name = test_name_base + str(counter)
    return counter, next_obj_name


def h5_fName_dict():
    h5_dSet_dict = {'exchange': 'recon_data_',
                    'workflow_cache': {'grayscale': 'gryscl_mod_',
                                       'binary_intermediate': 'binary_',
                                       'isolated_materials': ['material_',
                                                                'exterior'],
                                       'segmented_label_field': 'label_field_'},
                    'product': {'grayscale': 'gryscl_mod_',
                                 'binary_intermediate': 'binary_',
                                 'isolated_materials': ['material_',
                                                         'exterior'],
                                 'segmented_label_field': 'label_field_'}
                    }
    return h5_dSet_dict


# TODO Options for searching through the H5 Data set for either a particular
# group, to list data sets, groups, or map the entire file structure.
# Option 1 (painful) uses nexted for loops and a finite number of levels
# NOT QUITE COMPLETE
# file_obj = vol1
# h5_dict = {}
# for x in file_obj.keys():
    # print x
    # layer0_obj = file_obj[x]
    # if '.Dataset' in str(file_obj.get(x, getclass=True)): continue
    # if '.Group' in str(file_obj.get(x, getclass=True)):
        # for y in layer0_obj.keys():
            # print y
            # layer1_obj = layer0_obj[y]
            # if '.Dataset' in str(layer0_obj.get(y, getclass=True)):
                # if y == layer0_obj.keys()[0]:
                    # h5_dict[str(x)] = [str(y)]
                # else:
                    # h5_dict[str(x)].append(str(y))
            # if '.Group' in str(layer0_obj.get(y, getclass=True)):
                # if y == layer0_obj.keys()[0]:
                    # h5_dict[str(x)] = [{str(y): []}]
                # else:
                    # h5_dict[str(x)].append({str(y): []})
                # if len(layer1_obj.keys()) == 0: continue
                # for z in layer1_obj.keys():
                    # print z
                    # layer2_obj = layer1_obj[z]
                    # if '.Dataset' in str(layer1_obj.get(z, getclass=True)):
                        # #h5_dict[str(y)] = [str(z)]
                        # h5_dict[str(x)][str(y)].append(str(z))
                    # if '.Group' in str(layer1_obj.get(z, getclass=True)):
                        # h5_dict[str(x)][str(y)].append({str(z): []})

# Option 2:  Recursive search function.
# TODO: STILL NEEDS TO BE COMPLETED
# for x in group.keys():
    # print x
    # obj = group[x]
    # if isinstance(obj, h5py.Dataset):
    # if '.Group' in str(file_obj.get(x, getclass=True)):
        # for y in layer0_obj.keys():


# if '.Dataset' in vol1[layer0_obj].get(layer1_obj, getclass=True):

            # for z in layer1_obj.keys():
