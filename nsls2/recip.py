# Copyright (c) Brookhaven National Lab 2O14
# All rights reserved
# BSD License
# See LICENSE for full text
"""
This module is for functions and classes specific to reciprocal space
calculations.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np

def project_to_sphere(img, dist_sample, detector_center, pixel_size, wavelength, 
                      ROI = None, **kwargs):
    """
    Project the pixels on the 2D detector to the surface of a sphere.
    
    Parameters
    ----------
    img: ndarray
         2D detector image
    dist_sample: float
                 see keys_core
                 (mm)
    detector_center: 2 element float array
                     see keys_core
                     (pixels)
    pixel_size: 2 element float array
                see keys_core
                (mm)
    wavelength: float
                see keys_core
                (Angstroms)
    ROI: 4 element int array
           ROI defines a rectangular ROI for img
           ROI[0] == x_min
           ROI[1] == x_max
           ROI[2] == y_min
           ROI[3] == y_max
    **kwargs: dict
              Bucket for extra parameters from an unpacked dictionary
    
    Returns
    -------
    qi: 4 x N array of the coordinates in Q space (A^-1)
        Rows correspond to individual pixels
        Columns are (Qx, Qy, Qz, I)
    """
    
    if ROI is not None and len(ROI) == 4:
        # slice the image based on the desired ROI
        img=img[ROI[0]:ROI[1], ROI[2]:ROI[3]]
        
    # create the array of x indices
    arr_2d_x = np.zeros((img.shape[0], img.shape[1]), dtype=np.float)
    for x in range(img.shape[0]):
        arr_2d_x[x:x + 1] = x + 1 + ROI[0]

    # create the array of y indices
    arr_2d_y = np.zeros((img.shape[0], img.shape[1]), dtype=np.float)
    for y in range(img.shape[1]):
        arr_2d_y[:, y:y + 1] = y + 1 + ROI[2]

    # subtract the detector center 
    arr_2d_x -= detector_center[0]
    arr_2d_y -= detector_center[1]
    
    # convert the pixels into real-space dimensions
    arr_2d_x *= pixel_size[0]
    arr_2d_y *= pixel_size[1] 

    print("Image shape: {0}".format(img.shape))
    # define a new 4 x N array
    qi = np.zeros((4,) + (img.shape[0] * img.shape[1],))
    # fill in the x coordinates
    qi[0] = arr_2d_x.flatten()
    # fill in the y coordinates
    qi[1] = arr_2d_y.flatten()
    # set the z coordinate for all pixels to the distance from the sample to the detector
    qi[2].fill(dist_sample)
    # fill in the intensity values of the pixels
    qi[3] = img.flatten()
    # convert to an N x 4 array
    qi = qi.transpose()
    # compute the unit vector of each pixel
    qi[:,0:2] = qi[:,0:2]/np.linalg.norm(qi[:,0:2])
    # convert the pixel positions from real space distances into the reciprocal space
    # vector, Q
    Q = 4 * np.pi / wavelength * np.sin(np.arctan(qi[:,0:2]))
    # project the pixel coordinates onto the surface of a sphere of radius dist_sample
    qi[:,0:2] *= dist_sample
    # compute the vector from the center of the detector (i.e., the zero of reciprocal
    # space) to each pixel
    qi[:,2] -= dist_sample
    # compute the unit vector for each pixels position relative to the center of the 
    # detector, but now on the surface of a sphere
    qi[:,0:2] = qi[:,0:2]/np.linalg.norm(qi[:,0:2])
    # convert to reciprocal space
    qi[:,0:2] *= Q
    
    return qi
