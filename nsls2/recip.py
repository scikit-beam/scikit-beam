# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Redistribution and use in source and binary forms, with or without   #
# modification, are permitted provided that the following conditions   #
# are met:                                                             #
#                                                                      #
# * Redistributions of source code must retain the above copyright     #
#   notice, this list of conditions and the following disclaimer.      #
#                                                                      #
# * Redistributions in binary form must reproduce the above copyright  #
#   notice this list of conditions and the following disclaimer in     #
#   the documentation and/or other materials provided with the         #
#   distribution.                                                      #
#                                                                      #
# * Neither the name of the Brookhaven Science Associates, Brookhaven  #
#   National Laboratory nor the names of its contributors may be used  #
#   to endorse or promote products derived from this software without  #
#   specific prior written permission.                                 #
#                                                                      #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS  #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT    #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS    #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE       #
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,           #
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR   #
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)   #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,  #
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OTHERWISE) ARISING   #
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE   #
# POSSIBILITY OF SUCH DAMAGE.                                          #
########################################################################
"""
This module is for functions and classes specific to reciprocal space
calculations.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
import logging
logger = logging.getLogger(__name__)
import exceptions
import time
import gc
import operator
import ctrans


def project_to_sphere(img, dist_sample, detector_center, pixel_size,
                      wavelength, ROI=None, **kwargs):
    """
    Project the pixels on the 2D detector to the surface of a sphere.
    
    Parameters
    ==========
    img : ndarray
         2D detector image
    dist_sample : float
                 see keys_core
                 (mm)
    detector_center : 2 element float array
                     see keys_core
                     (pixels)
    pixel_size : 2 element float array
                see keys_core
                (mm)
    wavelength : float
                see keys_core
                (Angstroms)
    ROI : 4 element int array
           ROI defines a rectangular ROI for img
           ROI[0] == x_min
           ROI[1] == x_max
           ROI[2] == y_min
           ROI[3] == y_max
    **kwargs : dict
              Bucket for extra parameters from an unpacked dictionary


    Returns
    =======
    qi : 4 x N array of the coordinates in Q space (A^-1)
        Rows correspond to individual pixels
        Columns are (Qx, Qy, Qz, I)
        
    """

    if ROI is not None and len(ROI) == 4:
        # slice the image based on the desired ROI
        img = img[ROI[0]:ROI[1], ROI[2]:ROI[3]]
        
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
    # set the z coordinate for all pixels to
    # the distance from the sample to the detector
    qi[2].fill(dist_sample)
    # fill in the intensity values of the pixels
    qi[3] = img.flatten()
    # convert to an N x 4 array
    qi = qi.transpose()
    # compute the unit vector of each pixel
    qi[:, 0:2] = qi[:, 0:2]/np.linalg.norm(qi[:, 0:2])
    # convert the pixel positions from real space distances
    # into the reciprocal space
    # vector, Q
    Q = 4 * np.pi / wavelength * np.sin(np.arctan(qi[:, 0:2]))
    # project the pixel coordinates onto the surface of a sphere
    # of radius dist_sample
    qi[:, 0:2] *= dist_sample
    # compute the vector from the center of the detector
    # (i.e., the zero of reciprocal space) to each pixel
    qi[:, 2] -= dist_sample
    # compute the unit vector for each pixels position
    # relative to the center of the detector,
    #  but now on the surface of a sphere
    qi[:, 0:2] = qi[:, 0:2]/np.linalg.norm(qi[:, 0:2])
    # convert to reciprocal space
    qi[:, 0:2] *= Q
    
    return qi


def process_to_q(settingAngles, detSizeX, detSizeY, detPixSizeX,
                 detPixSizeY, detX0, detY0, detDis, waveLen, UBmat, istack):
    """
        
    This will procees the given images (certain scan) of
    the full set into receiprocal space (Q) (Qx, Qy, Qz, I)
    
    Parameters :
    ------------
    settingAngles : array
        six angles of the all the images
        
    detSizeX : int
        detector no. of pixels (size) in detector X-direction
        
    detSizeY : int
        detector no. of pixels (size) in detector Y-direction
        
    detPixSizeX : float
        detector pixel size in detector X-direction (mm)
        
    detPixSizeY : float
        detector pixel size in detector Y-direction (mm)
        
    detX0 : float
        detector X-coordinate of center for reference
        
    detY0 : float
        detector Y-coordinate of center for reference
        
    detDis : float
        detector distance from sample (mm)
        
    waveLen : float
        wavelength (Angstrom)
        
    UBmat : ndarray
        UB matrix (orientation matrix)
        
    istack : ndarray
        intensity array of the images
        
    Returns :
    --------
    totSet : ndarray
        (Qx, Qy, Qz, I) - HKL values and the intensity
        
    Optional :
    ----------
    frameMode = 4 : 'hkl'      : Reciproal lattice units frame.
    frameMode = 1 : 'theta'    : Theta axis frame.
    frameMode = 2 : 'phi'      : Phi axis frame.
    frameMode = 3 : 'cart'     : Crystal cartesian frame.
    
    """"
    
    ccdToQkwArgs = {}
    
    totSet = None
    gc.collect()
    # frameMode 4 : 'hkl'      : Reciproal lattice units frame.
    frameMode = 4
    
    if settingAngles is None:
        raise Exception(" No setting angles specified. ")
    
    # "---- Setting angle size :", settingAngles.shape
    # "---- CCD Size :", (detSizeX, detSizeY)
    
    #  **** Converting to Q   **************

    # starting time for the process
    t1 = time.time()

    # ctrans - c routines for fast data anlysis
    totSet = ctrans.ccdToQ(angles=settingAngles * np.pi / 180.0,
                           mode=frameMode,
                           ccd_size=(detSizeX, detSizeY),
                           ccd_pixsize=(detPixSizeX, detPixSizeY),
                           ccd_cen=(detX0, detY0),
                           dist=detDis,
                           wavelength=waveLen,
                           UBinv=np.matrix(UBmat).I,
                           **ccdToQkwArgs)
    # ending time for the process
    t2 = time.time()
                           
    #    "---- DONE (Processed in %f seconds)", %(t2 - t1)
    #    "---- Setsize is %d", %totSet.shape[0]
    totSet[:, 3] = np.ravel(istack)
                           
    return totSet


def process_grid(totSet, Qmin=None, Qmax=None, dQN=None):
    """
        
    This function will process the set of
    (Qx, Qy, Qz, I) values and grid the data
        
    Prameters :
    -----------
    totSet : ndarray
        (Qx, Qy, Qz, I) - HKL values and the intensity
        
    Returns :
    ---------
    gridData : ndarray
        intensity grid
        
    gridStd : ndarray
        standard devaiation grid
        
    gridOccu : int
        occupation of the grid
        
    gridOut : int
        No. of data point outside of the grid
        
    emptNb : int
        No. of values zero in the grid
        
    gridbins : int
        No. of bins in the grid
        
    Optional :
    ----------
    Qmin : ndarray
        minimum values of the cuboid [Qx, Qy, Qz]_min
    Qmax : ndarray
        maximum values of the cuboid [Qx, Qy, Qz]_max
    dQN  : ndarray
        No. of grid parts (bins)     [Nqx, Nqy, Nqz]
        
    """
    
    if totSet is None:
        raise Exception("No set of (Qx, Qy, Qz, I). Cannot process grid.")
    
    # prepare min, max,... from defaults if not set
    if Qmin is None:
        Qmin = np.array([totSet[:, 0].min(), totSet[:, 1].min(),
                         totSet[:, 2].min()])
    if Qmax is None:
        Qmax = np.array([totSet[:, 0].max(), totSet[:, 1].max(),
                         totSet[:, 2].max()])
    if dQN is None:
        dQN = [100, 100, 100]

    #            3D grid of the data set
    #             *** Gridding Data ****
    
    # staring time for griding
    t1 = time.time()

    # ctrans - c routines for fast data anlysis
    gridData, gridOccu, gridStd, gridOut = ctrans.grid3d(totSet, Qmin, Qmax, dQN, norm=1)

    # ending time for the griding
    t2 = time.time()
    
    # "---- DONE (Processed in %f seconds)" % (t2 - t1)
    # No. of bins in the grid
    gridbins = gridData.size

    # No. of values zero in the grid
    emptNb = (gridOccu == 0).sum()
    
    if gridOut != 0:
        print "---- Warning : There are %.2e points outside the grid" % gridOut
        print " (%.2e bins in the grid)" % gridData.size
    if emptNb:
        print "---- Warning : There are %.2e values zero in the grid" % emptNb
    
    return gridData, gridOccu, gridStd, gridOut, emptNb, gridbins


def get_grid_mesh(Qmin, Qmax, dQN):
    """
        
    This function returns the X, Y and Z coordinates of the grid as 3d
    arrays. (Return the grid vectors as a mesh.
    Parameters :
    -----------
    Qmin : ndarray
        minimum values of the cuboid [Qx, Qy, Qz]_min
        
    Qmax : ndarray
        maximum values of the cuboid [Qx, Qy, Qz]_max
        
    dQN  : ndarray
        No. of grid parts (bins) [Nqx, Nqy, Nqz]
        
    Returns :
    --------
    X : array
        X co-ordinate of the grid
        
    Y : array
        Y co-ordinate of the grid
        
    Z : array
        Z co-ordinate of the grid
        
        
    Example :
    ---------
    These values can be used for obtaining the coordinates of each voxel.
    For instance, the position of the (0,0,0) voxel is given by
        
        x = X[0,0,0]
        y = Y[0,0,0]
        z = Z[0,0,0]
        
    """
    
    grid = np.mgrid[0:dQN[0], 0:dQN[1], 0:dQN[2]]
    r = (Qmax - Qmin) / dQN
    
    X = grid[0] * r[0] + Qmin[0]
    Y = grid[1] * r[1] + Qmin[1]
    Z = grid[2] * r[2] + Qmin[2]
    
    return X, Y, Z
