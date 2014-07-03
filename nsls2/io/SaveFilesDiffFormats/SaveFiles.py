# Copyright (c) Brookhaven National Lab 2O14
# All rights reserved
# BSD License
# See LICENSE for full text
"""
This module is for Saving the X-ray diffraction
intencites in differnet file formats
"""


import numpy as np
import scipy.io
import os


def saveChi(tth, intensity, err, filename):
    '''
    Save diffraction intensities in .chi file format
    Parameters :
    --------------
    tth :
        2(theta) values or (Q values)
    intensity :
               intensisity values
    err :
          uncertainiy
    path : string
           directory to save the chi files
    filename : string
               filename(could be full path) of the .tif file
    Returns :
    ---------
    Saved file of diffraction intensities in .chi file format
    '''
    filebase = os.path.splitext(os.path.split(filename)[1])[0]
    filepath = filebase + '.chi'
    f = open(filepath, 'wb')
    np.savetxt(f, (tth, intensity, err), newline='\n')
    f.close()
    return


def saveDat(tth, intensity, err, filename):
    '''
    Save diffraction intensities in .dat file format
    Parameters :
    --------------
    tth :
          2(theta) values or (Q values)
    intensity :
               intensisity values
    err :
          uncertainiy
    path : string
           directory to save the chi files
    filename : string
               filename(could be full path) of the .tif file
    Returns :
    ---------
    Saved file of diffraction intensities in .dat file format
    '''
    filebase = os.path.splitext(os.path.split(filename)[1])[0]
    filepath = filebase + '.dat'
    f = open(filepath, 'wb')
    np.savetxt(f, (tth, intensity, err), newline='\n')
    f.close()
    return


def saveXYE(tth, intensity, err, filepath, filebase):
    '''
    Save diffraction intensities in .xye file format
    '''
    pass


def saveGSAS(tth, intensity, err, filepath, filename):
    '''
    Save diffraction intensities in .gsas file format
    '''
    pass
