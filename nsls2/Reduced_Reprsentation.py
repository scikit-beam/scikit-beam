# Copyright (c) Brookhaven National Lab 2O14
# All rights reserved
# BSD License
# See LICENSE for full text
""" This Modeule is for
    Statistics or Reduced Representation of 2D images 
    (Mean, Total Intensity, Standard Deviation ) """


import numpy as np

def RR1Choice(image_data):
    '''
    Parameters
    ----------
    image_data : ndarray
    MxN array of data
    -------
    Reduced Representation Choices
    --------------------------------
    rrm : Mean
    rrt : Total Intensity
    rrs : Standard Deviation
    '''
    rrm = np.mean(image_data)
    rrt = np.sum(image_data)
    rrs = np.sum(image_data)
    return rrm, rst, rrs


