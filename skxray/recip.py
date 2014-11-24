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


def hkl_to_q(hkl_arr):
    """
    This module compute the reciprocal space (q) values from known HKL array
    for each pixel of the detector for all the images

    Parameters
    ----------
    hkl_arr : ndarray
        (Qx, Qy, Qz) - HKL array
        shape is [num_images * num_rows * num_columns][3]

    Returns
    -------
    q_val : ndarray
        Reciprocal values for each pixel for all images
        shape is [num_images * num_rows * num_columns]
    """

    return np.linalg.norm(hkl_arr, axis=1)


def q_rings(num_qs, first_q, delta_q, q_val, step_q=None):
    """
    This will find the indices of the required Q rings, find the bin
    edges of the Q rings, and count the number of pixels in each Q ring.

    Parameters
    ----------
    num_qs : int
        number of Q rings

    first_q : float
        Q value of the first Q ring

    delta_q : float
        thickness of the Q ring

    q_val : ndarray
        Q space values for each pixel in the detector
        shape is [detector_size[0]*detector_size[1]][1] or
        (Qx, Qy, Qz) - HKL values
        shape is [detector_size[0]*detector_size[1]][3]

    step_q : float, optional
        step value for the next Q ring from the end of the previous Q ring

    Returns
    -------
    q_inds : ndarray
        indices of the Q values for the required rings

    q_ring_val : ndarray
        edge values of each Q ring
        shape is [num_qs][2]

    num_pixels : ndarray
        number of pixels in each Q ring
    """

    if (delta_q < 0):
        raise ValueError("delta_q(thickness of the Q ring has to be positive")

    if (q_val.ndim == 1):
        q_values = q_val
    elif (q_val.ndim == 2):
        q_values = np.ravel(q_val)
    else:
        raise ValueError("Q space values for each pixel in the detector has"
                         " to be specified")

    if (step_q is None):
        # last Q ring edge value
        last_q = first_q + num_qs*(delta_q)
        # edges of all the Q rings
        q_r = np.linspace(first_q, last_q, num=(num_qs+1))

        # indices of Q rings
        q_inds = np.digitize(q_values, np.array(q_r))
        for i, item in enumerate(q_inds):
            if (item > num_qs):
                q_inds[i] = 0

        # Edge values of each Q rings
        q_ring_val = []

        count = 0
        for i in range(0, num_qs):
            if (i < (num_qs)):
                q_ring_val.append(q_r[count])
                q_ring_val.append(q_r[count + 1])
            else:
                q_ring_val.append(q_r[num_qs-1])
            count += 1
        q_ring_val = np.asarray(q_ring_val)

    else:
        if (step_q < 0):
            raise ValueError("step_q(step value for the next Q ring from"
                         " the end of the previous ring) has to be positive ")

        #  when there is a step between Q rings find the edge values of Q rings
        q_ring_val = first_q + np.r_[0, np.cumsum(np.tile([delta_q, step_q], num_qs))][:-1]

        # indices of Q rings
        q_inds = np.digitize(q_values, np.array(q_ring_val))

        # to discard every-other bin and set the discarded bins indices to 0
        q_inds[q_inds % 2 == 0] = 0
        # change the indices of odd number of rings
        indx = q_inds > 0
        q_inds[indx] = (q_inds[indx] + 1) // 2

    q_ring_val = np.array(q_ring_val)
    q_ring_val = q_ring_val.reshape(num_qs, 2)

    # number of pixels in each  Q ring
    num_pixels = np.bincount(q_inds, minlength=(num_qs+1))
    num_pixels = num_pixels[1:]

    return q_inds, q_ring_val, num_pixels
