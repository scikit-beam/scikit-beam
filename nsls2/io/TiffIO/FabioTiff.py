# Copyright (c) Brookhaven National Lab 2O14
# All rights reserved
# BSD License
# See LICENSE for full text
"""
    This module is for read the .tiff file(load_tif) using FabIO
    
    """

import os
import numpy as np
import fabio


def load_tif(file):
    """
    Parameters
    ----------
    file_name: string
               Complete path to the file to be loaded into memory
    Returns
    -------
    output: NxN ndarray
            Returns a numpy array of the tiff file
    """
    image = fabio.open(file)
    image_data = image.data
    return image_data


