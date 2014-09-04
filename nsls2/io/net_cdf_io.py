# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Nov. 2013
"""
This module contains fileIO operations and file conversion for the image 
processing tool kit in the NSLS-II data analysis software package. 
The functions included in this module focus on reading and writing
netCDF files. This is the file format used by Mark Rivers for 
x-ray computed microtomography data collected at Argonne National Laboratory,
Sector 13BMD, GSECars.
"""
"""
REVISION LOG: (FORMAT: "PROGRAMMER INITIALS: DATE -- RECORD")
-------------------------------------------------------------
GCI: 5/13/14 -- Added load function for netCDF files. Specifically for loading 
    data sets acquired at the APS Sector 13 beamline.
GCI: 8/1/14 -- Updating documentation to detail required dependencies for 
    netCDF file IO. Without these required dependencies these functions
    will not work.
"""

import numpy as np
from netCDF4 import Dataset


def load_netCDF(file_name, 
                data_type = None):
    """
    This function loads the specified netCDF file format data set (e.g.*.volume 
    APS-Sector 13 GSECARS extension) file into a numpy array for further analysis.
    
    Required Dependencies
    ---------------------
    netcdf4 : Python/numpy interface to the netCDF ver. 4 library
        Package name: netcdf4-python
        Install from: https://github.com/Unidata/netcdf4-python
    
    numpy
    
    Cython -- optional
    
    HDF5 C library version 1.8.8 or higher
        Install from:  ftp://ftp.hdfgroup.org/HDF5/current/src
        Be sure to build with '--enable-hl --enable-shared'.
    
    netCDF-4 C library 
        Install from: 
            ftp://ftp.unidata.ucar.edu/pub/netcdf. Version 4.1.1 or higher
        Be sure to build with '--enable-netcdf-4 --enable-shared', and set 
        CPPFLAGS="-I $HDF5_DIR/include" and LDFLAGS="-L $HDF5_DIR/lib", where 
        $HDF5_DIR is the directory where HDF5 was installed. 
        If you want OPeNDAP support, add '--enable-dap'. 
        If you want HDF4 SD support, add '--enable-hdf4' and add the location 
        of the HDF4 headers and library to CPPFLAGS and LDFLAGS.
    
    
    Parameters
    ----------
    file_name : string
        Complete path to the file to be loaded into memory
    
    data_type : integer
        Z-axis array dimension as an integer value

    
    Returns
    -------
    data : ndarray
        ndarray containing the image data contained in the netCDF file.
        The image data is scaled using the scale factor defined in the
        netCDF metadata, if a scale factor was recorded during data 
        acquisition or reconstruction. If a scale factor is not present, 
        then a default value of 1.0 is used.
    
    tmp_dict : dict
        Dictionary containing all metadata contained in the netCDF file.
        This metadata contains data collection, and experiment information
        as well as values and variables pertinent to the image data.
    """
    
    def _md_dict_cleanup(imported_dict):
        #dictioary cleanup
        

    def _md_dict_convert(cleanded_dict):
#dictionary parameter assignment for export
#names need to be registered and correspond directly to those entered in core.py
    src_file = Dataset(file_name)
    data = src_file.variables['VOLUME']
    tmp_dict = src_file.__dict__
    try:
        scale_value = data.scale_factor
        print 'Image data values adjusted by the recorded scale factor.'
    except:
        print 'the scale factor is non existent'
        scale_value = 1.0
    data=data/scale_value
    return data, tmp_dict 
   
