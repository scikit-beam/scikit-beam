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
from .core import verbosedict
import sys
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
                 calibrated_center, dist_sample, wavelength, ub,
                 frame_mode=None):
    """
    This will compute the hkl values for all pixels in a shape specified by
    detector_size.

    Parameters
    ----------
    setting_angles : ndarray
        six angles of all the images - Required shape is [num_images][6] and
        required type is something that can be cast to a 2D numpy array
        Angle order: delta, theta, chi, phi, mu, gamma (degrees)

    detector_size : tuple
        2 element tuple defining the number of pixels in the detector. Order is
        (num_columns, num_rows)

    pixel_size : tuple
        2 element tuple defining the size of each pixel in mm. Order is
        (column_pixel_size, row_pixel_size).  If not in mm, must be in the same
        units as `dist_sample`

    calibrated_center : tuple
        2 element tuple defining the center of the detector in pixels. Order
        is (column_center, row_center)(x y)

    dist_sample : float
        distance from the sample to the detector (mm). If not in mm, must be
        in the same units as `pixel_size`

    wavelength : float
        wavelength of incident radiation (Angstroms)

    ub : ndarray
        UB matrix (orientation matrix) 3x3 matrix

    frame_mode : str, optional
        Frame mode defines the data collection mode and thus the desired
        output from this function. Defaults to hkl mode (frame_mode=4)
        'theta'    : Theta axis frame.
        'phi'      : Phi axis frame.
        'cart'     : Crystal cartesian frame.
        'hkl'      : Reciprocal lattice units frame.
        See the `process_to_q.frame_mode` attribute for an exact list of
        valid options.

    Returns
    -------
    hkl : ndarray
        (Qx, Qy, Qz) - HKL values
        shape is [num_images * num_rows * num_columns][3]

    Notes
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
    # set default frame_mode
    if frame_mode is None:
        frame_mode = 4
    else:
        str_to_int = verbosedict((k, j+1) for j, k
                                 in enumerate(process_to_q.frame_mode))
        frame_mode = str_to_int[frame_mode]
    # ensure the ub matrix is an array
    ub = np.asarray(ub)
    # ensure setting angles is a 2-D
    setting_angles = np.atleast_2d(setting_angles)
    if setting_angles.ndim != 2:
        raise ValueError('setting_angles is expected to be a 2-D array with'
                         ' dimensions [num_images][num_angles]. You provided '
                         'an array with dimensions {0}'
                         ''.format(setting_angles.shape))
    if setting_angles.shape[1] != 6:
        raise ValueError('It is expected that there should be six angles in '
                         'the setting_angles parameter. You provided {0}'
                         ' angles.'.format(setting_angles.shape[1]))
    #  *********** Converting to Q   **************

    # starting time for the process
    t1 = time.time()

    # ctrans - c routines for fast data analysis
    hkl = ctrans.ccdToQ(angles=setting_angles * np.pi / 180.0,
                        mode=frame_mode,
                        ccd_size=(detector_size),
                        ccd_pixsize=(pixel_size),
                        ccd_cen=(calibrated_center),
                        dist=dist_sample,
                        wavelength=wavelength,
                        UBinv=np.matrix(ub).I)
                        # **kwargs)

    # ending time for the process
    t2 = time.time()
    logger.info("Processing time for {0} {1} x {2} images took {3} seconds."
                "".format(setting_angles.shape[0], detector_size[0],
                          detector_size[1], (t2-t1)))
    return hkl[:, :3]

# Assign frame_mode as an attribute to the process_to_q function so that the
# autowrapping knows what the valid options are
process_to_q.frame_mode = ['theta', 'phi', 'cart', 'hkl']


def q_roi(num_rois, roi_data, q_val, detector_size):
    """
    This module will find the indices of the required Q shape and count the
     number of pixels in that shape

     Parameter
     --------

    num_rois: int
        number of region of interests(roi)

    co_or: ndarray
        co-ordinates of roi's

    q_val:
        Q space values for each pixel in the detector
        shape is [detector_size[0]*detector_size[1]][1] or
        (Qx, Qy, Qz) - HKL values
        shape is [detector_size[0]*detector_size[1]][3]

    detector_size:

    Returns
    -------
    q_inds : ndarray
        indices of the Q values for the required shape

    num_pixels : ndarray
        number of pixels in certain Q shape

    """

    if (q_val.ndim == 1):
        q_values = q_val
    elif (q_val.ndim == 3):
        q_values = np.sqrt(q_val[:, 0]**2 + q_val[:, 1]**2 + q_val[:, 2]**2)
    else:
        raise ValueError("Either HKL values(Qx, Qy, Qz) or Q space values"
                            " for each pixel in the detector has to be specified")

    q_mesh = np.zeros((detector_size[0], detector_size[1]))

    for i in (0, num_rois):
        x_coor, y_coor = roi_data[0], roi_data[1]
        x_val = roi_data[3]
        y_val = roi_data[4]
        if ((x_val + x_coor)< detector_size[0] and (y_val + y_coor) < detector_size[1]):
            q_mesh[x_coor: x_coor + x_val, y_coor: y_coor + y_val] = np.ones((x_val, y_val))*i
        else:
            raise ValueError("Could not broadcast input array from shape ({0},{1}) into"
                             " ({2},{3})".format(x_val, y_val,(detector_size[0] - x_coor - x_val),
                                                (detector_size[1] - y_coor - y_val)))
