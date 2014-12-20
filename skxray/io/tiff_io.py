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
import tifffile

def load_RAW(file_name, 
             z, 
             y, 
             x):
    """
    This function loads the specified RAW file format data set (.raw, or .volume 
    extension) file into a numpy array for further analysis.
    
    Parameters
    ----------
    file_name : string
	Complete path to the file to be loaded into memory
    
    z : integer
	Z-axis array dimension as an integer value

    y : integer
	Y-axis array dimension as an integer value

    x : integer
	X-axis array dimension as an integer value

    Returns
    -------
    output : NxN or NxNxN ndarray
	Returns the loaded data set as a 32-bit float numpy array

    Example
    -------
    vol = fileops.load_RAW('file_path', 520, 695, 695)
    """
    src_volume = np.empty((z,
                           y,
                           x), np.float32)
    print('loading ' + file_name)
    src_volume.data[:] = open(file_name).read() #sample file is now loaded
    target_var = src_volume[:,:,:]
    print 'Volume loaded successfully'
    return target_var


def read_tiff(file_name):
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
    file_name : str
	Complete path to the file to be loaded into memory

    Returns
    -------
    output : array
	Returns a numpy array of the same data type as the original tiff file

    Example
    -------
    vol = fileops.load_tif('file_path')

    """
    target_var = tifffile.imread(file_name)
    return target_var


def save_tiff(file_name,
                   data):
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
    tifffile.imsave(file_name, 
                    data)


