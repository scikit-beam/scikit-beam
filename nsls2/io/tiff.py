# Copyright (c) Brookhaven National Lab 2O14
# All rights reserved
# BSD License
# See LICENSE for full text
"""
    This module is for read the image(.tiff,.edf, .img, etc..) files(load_img) using FabIO
    
    """

import os
import numpy as np
import fabio
import Image


def load_img(file):
    """
    Parameters
    ----------
    file_name : string
                Complete path to the image file to be loaded into memory
    Returns
    -------
    output : NxN ndarray
             Returns a numpy array of the image file
             
    Note
    ----
    FabIO is an I/O library for images produced by 2D X-ray detectors.
    FabIO support images detectors from a dozen of companies (including Mar, 
    Dectris, ADSC, Hamamatsu, Oxford,.),for a total of 20 different file formats 
    (like CBF, EDF, TIFF, ...) and offers an unified interface to their headers 
    (as a python dictionary)and datasets (as a numpy ndarray of integers or floats)
    https://github.com/kif/fabio
    
    """
    image = fabio.open(file)
    return image.data


def write_tif(image, filname):
    """
    Save the image data as a .tif file
    Parameters :
    -------------
    image : N x N array
            image data
    filename : string
               filename to save the tiff file

    Returns:
    ---------
    output : string
             Saved tif file
    """
    im = Image.fromarray(image)
    im.save(filename)
    return
