# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Nov. 2013
"""
RAW (.raw and .volume)
"""

import numpy as np

def read_RAW(file_name,
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
