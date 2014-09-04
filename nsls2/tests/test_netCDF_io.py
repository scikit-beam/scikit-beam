# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Sept. 2014
"""
This module contains test functions for the file-IO functions
for reading and writing data sets using the netCDF file format.

The files read and written using this function are assumed to 
conform to the format specified for x-ray computed microtomorgraphy
data collected at Argonne National Laboratory, Sector 13, GSECars.
"""

import numpy as np
import six
import net_cdf_io as ncd

test_data = ../../../test_data/file_io/netCDF/'tst_netCDF_recon.volume'

def test_net_cdf_io(test_data):
    data, md_dict = ncd.load_netCDF(test_data)
    if md_dict['operator'] != 'Iltis':
        raise Exception

