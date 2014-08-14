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
import time

try:
    import src.ctrans as ctrans
except ImportError:
    try:
        import ctrans
    except ImportError:
        ctrans = None


def project_to_sphere(img, dist_sample, calibrated_center, pixel_size,
                      wavelength, ROI=None, **kwargs):
    """
    Project the pixels on the 2D detector to the surface of a sphere.

    Parameters
    ----------
    img : ndarray
        2D detector image

    dist_sample : float
        see keys_core  (mm)

    calibrated_center : 2 element float array
        see keys_core (pixels)

    pixel_size : 2 element float array
        see keys_core (mm)

    wavelength : float
        see keys_core (Angstroms)

    ROI : 4 element int array
        ROI defines a rectangular ROI for img
        ROI[0] == x_min
        ROI[1] == x_max
        ROI[2] == y_min
        ROI[3] == y_max

    **kwargs : dict
        Bucket for extra parameters from an unpacked dictionary

    Returns
    -------
    Bucket for extra parameters from an unpacked dictionary

    qi : 4 x N array of the coordinates in Q space (A^-1)
        Rows correspond to individual pixels
        Columns are (Qx, Qy, Qz, I)

    Raises
    ------
    ValueError
        Possible causes:
            Raised when the ROI is not a 4 element array

    ValueError
        Possible causes:
            Raised when ROI is not specified
    """

    if ROI is not None:
        if len(ROI) == 4:
            # slice the image based on the desired ROI
            img = np.meshgrid(img[ROI[0]:ROI[1]], img[ROI[2]:ROI[3]],
                              sparse=True)
        else:
            raise ValueError(" ROI has to be 4 element array : len(ROI) = 4")
    else:
        raise ValueError(" No ROI is specified ")

    # create the array of x indices
    arr_2d_x = np.zeros((img.shape[0], img.shape[1]), dtype=np.float)
    for x in range(img.shape[0]):
        arr_2d_x[x:x + 1] = x + 1 + ROI[0]

    # create the array of y indices
    arr_2d_y = np.zeros((img.shape[0], img.shape[1]), dtype=np.float)
    for y in range(img.shape[1]):
        arr_2d_y[:, y:y + 1] = y + 1 + ROI[2]

    # subtract the detector center
    arr_2d_x -= calibrated_center[0]
    arr_2d_y -= calibrated_center[1]

    # convert the pixels into real-space dimensions
    arr_2d_x *= pixel_size[0]
    arr_2d_y *= pixel_size[1]

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


def process_to_q(setting_angles, detector_size, pixel_size,
                 calibrated_center, dist_sample, wavelength, ub_mat):
    """
    This will procees the given images (certain scan) of
    the full set into receiprocal(Q) space, (Qx, Qy, Qz)

    Parameters
    ----------
    setting_angles : Nx6 array
        six angles of the all the images
        delta, theta, chi, phi, mu, gamma

    detector_size : tuple
        see keys_core (pixel)

    pixel_size : tuple
        see keys_core (mm)

    calibrated_center : tuple
        see key_core (mm)

    dist_sample : float
        see keys_core (mm)

    wavelength : float
        see keys_core (Angstroms)

    ub_mat : 3x3 array
        UB matrix (orientation matrix)

    Returns
    -------
    tot_set : Nx3 array
        (Qx, Qy, Qz) - HKL values

    Raises
    ------
    ValueError
        Possible causes:
            Raised when the diffractometer six angles of
            the images are not specified

    Note
    -----
    Six angles of an image: (delta, theta, chi, phi, mu, gamma )
    These axes are defined according to the following references.

    References: text [1]_, text [2]_

    .. [1] M. Lohmeier and E.Vlieg, "Angle calculations for a six-circle
       surface x-ray diffractometer," J. Appl. Cryst., vol 26, pp 706-716,
       1993.

    .. [2] E. Vlieg, "A (2+3)-Type surface diffractometer: Mergence of the
       z-axis and (2+2)-Type geometries," J. Appl. Cryst., vol 31, pp 198-203,
       1998.

    """
    ccdToQkwArgs = {}

    tot_set = None

    # frame_mode = 1 : 'theta'    : Theta axis frame.
    # frame_mode = 2 : 'phi'      : Phi axis frame.
    # frame_mode = 3 : 'cart'     : Crystal cartesian frame.
    # frame_mode = 4 : 'hkl'      : Reciprocal lattice units frame.
    frame_mode = 4

    if setting_angles is None:
        raise ValueError(" No six angles specified. ")

    #  *********** Converting to Q   **************

    # starting time for the process
    t1 = time.time()

    # ctrans - c routines for fast data analysis
    tot_set = ctrans.ccdToQ(angles=setting_angles * np.pi / 180.0,
                           mode=frame_mode,
                           ccd_size=(detector_size),
                           ccd_pixsize=(pixel_size),
                           ccd_cen=(calibrated_center),
                           dist=dist_sample,
                           wavelength=wavelength,
                           UBinv=np.matrix(ub_mat).I,
                           **ccdToQkwArgs)

    # ending time for the process
    t2 = time.time()
    logger.info("--- Done processed in %f seconds", (t2-t1))

    return tot_set[:, :3]


def process_grid(tot_set, i_stack, q_min=None, q_max=None, dqn=None):
    """
    This function will process the set of HKL
    values and the image stack and grid the image data

    Prameters
    ---------
    tot_set : Nx3 array
        (Qx, Qy, Qz) - HKL values

    istack : Nx1
        intensity array of the images

    q_min : ndarray, optional
        minimum values of the voxel[Qx, Qy, Qz]_min

    q_max : ndarray, optional
        maximum values of the voxel [Qx, Qy, Qz]_max

    dqn : ndarray, optional
        No. of grid parts (bins) [Nqx, Nqy, Nqz]

    Returns
    -------
    grid_data : ndarray
        intensity grid

    grid_std : ndarray
        standard deviation grid

    grid_occu : ndarray
        occupation of the grid

    grid_out : int
        No. of data point outside of the grid

    empt_nb : int
        No. of values zero in the grid

    grid_bins : int
        No. of bins in the grid

    Raises
    ------
    ValueError
        Possible causes:
            Raised when the HKL values are not provided
    """

    if tot_set is None:
        raise ValueError(" No set of (Qx, Qy, Qz). Cannot process grid. ")

    # creating (Qx, Qy, Qz, I) Nx4 array - HKL values and Intensity
    # getting the intensity value for each pixel
    tot_set = np.insert(tot_set, 3, np.ravel(i_stack), axis=1)

    # prepare min, max,... from defaults if not set
    if q_min is None:
        q_min = np.array([tot_set[:, 0].min(), tot_set[:, 1].min(),
                         tot_set[:, 2].min()])
    if q_max is None:
        q_max = np.array([tot_set[:, 0].max(), tot_set[:, 1].max(),
                         tot_set[:, 2].max()])
    if dqn is None:
        dqn = [100, 100, 100]

    #            3D grid of the data set
    #             *** Griding Data ****

    # starting time for griding
    t1 = time.time()

    # ctrans - c routines for fast data analysis
    (grid_data, grid_occu,
        grid_std, grid_out) = ctrans.grid3d(tot_set, q_min, q_max, dqn, norm=1)

    # ending time for the griding
    t2 = time.time()
    logger.info("--- Done processed in %f seconds", (t2-t1))

    # No. of bins in the grid
    grid_bins = grid_data.size

    # No. of values zero in the grid
    empt_nb = (grid_occu == 0).sum()

    if grid_out:
        logger.deug("There are %.2e points outside the grid ", grid_out)
    logger.debug("There are %2e bins in the grid ", grid_data.size)
    if empt_nb:
        logger.debug("There are %.2e values zero in th grid ", empt_nb)

    return grid_data, grid_occu, grid_std, grid_out, empt_nb, grid_bins
