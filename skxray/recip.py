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



def q_no_step_val(img_dim, calibrated_center, num_qs,
               first_q, delta_q):
    """
    This will provide the indices of the required q rings,
    find the bin edges of the q rings, and count the number
    of pixels in each q ring, and pixels list for the required
    q rings when there is no step value between rings.

    Parameters
    ----------
    q_val : ndarray
        q space values for each pixel in the detector
        shape is ([detector_size[0]*detector_size[1]], ) or
        ([detector_size[0]*detector_size[1]], 1)

    num_qs : int
        number of q rings

    first_q : float
        q value of the first q ring

    delta_q : float
        thickness of the q ring

    Returns
    -------
    q_ring_val : ndarray
        edge values of the required q rings

    q_inds : ndarray
        indices of the q values for the required rings

    """
    q_values = _grid_values(img_dim, calibrated_center)

    # last Q ring edge value
    last_q = first_q + num_qs*(delta_q)

    # edges of all the Q rings
    q_r = np.linspace(first_q, last_q, num=(num_qs+1))

    # indices of Q rings
    q_inds = np.digitize(q_values, np.array(q_r))
    # discard the indices greater than number of Q rings
    q_inds[q_inds > num_qs] = 0

    # Edge values of each Q rings
    q_ring_val = []

    for i in range(0, num_qs):
        if i < num_qs:
            q_ring_val.append(q_r[i])
            q_ring_val.append(q_r[i + 1])
        else:
            q_ring_val.append(q_r[num_qs-1])

    q_ring_val = np.asarray(q_ring_val)

    (q_inds, q_ring_val, num_pixels, pixel_list,
     all_pixels) = _process_q_rings(num_qs, img_dim, q_ring_val, q_inds)

    return q_inds, q_ring_val, num_pixels, all_pixels, pixel_list


def q_step_val(img_dim, calibrated_center, num_qs,
               first_q, delta_q, *args):
    """
    This will provide the indices of the required q rings,
    find the bin edges of the q rings, and count the number
    of pixels in each q ring, and pixels list for the required
    q rings when there is a step value between rings.
    Step value can be same or different steps between
    each q ring.

    Parameters
    ----------
    img_val : tuple

    num_qs : int
        number of q rings

    first_q : float
        q value of the first q ring

    delta_q : float
        thickness of the q ring

    *args : tuple
        step value for the next q ring from the end of the previous
        q ring. same step - same step values between q rings (one value)
        different steps - different step value between q rings (provide
        step value for each q ring eg: 6 rings provide 5 step values)

    Returns
    -------
    q_ring_val : ndarray
        edge values of q the required rings

    q_inds : ndarray
        indices of the q values for the required rings
    """

    q_values = _grid_values(img_dim, calibrated_center)

    q_ring_val = []

    for arg in args:
        if arg < 0:
            raise ValueError("step_q(step value for the next Q ring from the "
                             "end of the previous ring) has to be positive ")

    if len(args) == 1:
        #  when there is a same values of step between q rings
        #  the edge values of q rings will be
        q_ring_val = first_q + np.r_[0, np.cumsum(np.tile([delta_q,
                                                           float(args[0])],
                                                          num_qs))][:-1]
    else:
        # when there is a different step values between each q ring
        #  edge values of the q rings will be
        if len(args) == (num_qs-1):
            q_ring_val.append(first_q)
            for arg in args:
                q_ring_val.append(q_ring_val[-1] + delta_q)
                q_ring_val.append(q_ring_val[-1] + float(arg))
            q_ring_val.append(q_ring_val[-1] + delta_q)
        else:
            raise ValueError("Provide step value for each q ring ")

    # indices of Q rings
    q_inds = np.digitize(q_values, np.array(q_ring_val))

    # to discard every-other bin and set the discarded bins indices to 0
    q_inds[q_inds % 2 == 0] = 0
    # change the indices of odd number of rings
    indx = q_inds > 0
    q_inds[indx] = (q_inds[indx] + 1) // 2

    (q_inds, q_ring_val, num_pixels, pixel_list,
     all_pixels) = _process_q_rings(num_qs, img_dim, q_ring_val, q_inds)

    return q_inds, q_ring_val, num_pixels, all_pixels, pixel_list


def _grid_values(img_dim, calibrated_center):
    """
    Parameters
    ----------
    img_dim: tuple
        shape of the image (detector X and Y direction)
        shape is [detector_size[0], detector_size[1]])

    calibarted_center : tuple
        defining the (x y) center of the detector (mm)

    """
    xx, yy = np.mgrid[:img_dim[0], :img_dim[1]]
    grid_values = np.ravel((xx - calibrated_center[0]) ** 2
                           + (yy - calibrated_center[1]) ** 2)

    return grid_values


def _process_q_rings(num_qs, q_val_shape, q_ring_val, q_inds):
    """
    This will find the indices of the required q rings, find the bin
    edges of the q rings, and count the number of pixels in each q ring,
    and pixels list for the required q rings.

    Parameters
    ----------
    num_qs : int
        number of q rings

    q_val_shape : tuple
        shape of the q space values(for each pixel in the detector,
        shape is [detector_size[0]*detector_size[1]], )

    q_ring_val : ndarray
        edge values of each q ring

    q_inds : ndarray
        indices of the q values for the required rings
        shape is ([detector_size[0]*detector_size[1]], )

    Returns
    -------
    q_inds : ndarray
        indices of the q values for the required rings
        (after discarding zero values from the shape
        ([detector_size[0]*detector_size[1]], )

    q_ring_val : ndarray
        edge values of each q ring
        shape is (num_qs, 2)

    num_pixels : ndarray
        number of pixels in each q ring

    all_pixels : int
        sum of pixels of all the required q rings

    pixel_list : ndarray
        pixel list for the required q rings
    """

    # find the pixel list
    w = np.where(q_inds > 0)
    grid = np.indices([q_val_shape[0], q_val_shape[1]])
    pixel_list = np.ravel(grid[1]*q_val_shape[0] + grid[0])[w]

    q_inds = q_inds[q_inds > 0]

    q_ring_val = np.array(q_ring_val)
    q_ring_val = q_ring_val.reshape(num_qs, 2)

    # number of pixels in each  Q ring
    num_pixels = np.bincount(q_inds, minlength=(num_qs+1))
    num_pixels = num_pixels[1:]

    # sum of pixels of all the required q rings
    all_pixels = sum(num_pixels)

    return q_inds, q_ring_val, num_pixels, all_pixels, pixel_list


def q_rings1(num_qs, q_val_shape, q_ring_val, q_inds):
    """
    This will find the indices of the required q rings, find the bin
    edges of the q rings, and count the number of pixels in each q ring,
    and pixels list for the required q rings.

    Parameters
    ----------
    num_qs : int
        number of q rings

    q_val_shape : tuple
        shape of the q space values(for each pixel in the detector,
        shape is [detector_size[0]*detector_size[1]], )

    q_ring_val : ndarray
        edge values of each q ring

    q_inds : ndarray
        indices of the q values for the required rings

    Returns
    -------
    q_inds : ndarray
        indices of the q values for the required rings

    q_ring_val : ndarray
        edge values of each q ring
        shape is (num_qs, 2)

    num_pixels : ndarray
        number of pixels in each q ring

    pixel_list : ndarray
        pixel list for the required q rings
    """

    # find the pixel list
    w = np.where(q_inds > 0)
    grid = np.indices([q_val_shape[0], q_val_shape[1]])
    pixel_list = np.ravel(grid[1]*q_val_shape[0] + grid[0])[w]

    q_ring_val = np.array(q_ring_val)
    q_ring_val = q_ring_val.reshape(num_qs, 2)

    # number of pixels in each  Q ring
    num_pixels = np.bincount(q_inds, minlength=(num_qs+1))
    num_pixels = num_pixels[1:]

    return q_inds, q_ring_val, num_pixels, pixel_list


def _validate_q1(q_values, delta_q):
    """
    Parameters
    ----------
    q_values : ndarray
        q space values for each pixel in the detector
        shape is ([detector_size[0]*detector_size[1]], ) or
        ([detector_size[0]*detector_size[1]], 1)

    delta_q : float
        thickness of the q ring

    Returns
    -------
     q_val : ndarray
        q space values for each pixel in the detector
        shape is ([detector_size[0]*detector_size[1]], )
    """

    if delta_q < 0:
        raise ValueError("delta_q(thickness of the"
                         " q ring has to be positive")

    q_val = np.asarray(q_values)

    if q_val.ndim == 1:
        q_values = q_val
    elif q_val.ndim == 2:
        q_values = np.ravel(q_val)
    else:
        raise ValueError("q space values for each pixel in the detector"
                         " has to be specified")
    return q_val


def q_no_step_val1(q_val, num_qs, first_q, delta_q, q_values):
    """
    This will provide q rings edge values when there is no step value
    between rings.

    Parameters
    ----------
     q_val : ndarray
        q space values for each pixel in the detector
        shape is ([detector_size[0]*detector_size[1]], ) or
        ([detector_size[0]*detector_size[1]], 1)

    num_qs : int
        number of q rings

    first_q : float
        q value of the first q ring

    delta_q : float
        thickness of the q ring

    Returns
    -------
    q_ring_val : ndarray
        edge values of the required q rings

    q_inds : ndarray
        indices of the q values for the required rings
    """

    q_values = _validate_q1(q_val, delta_q)

    # last Q ring edge value
    last_q = first_q + num_qs*(delta_q)

    # edges of all the Q rings
    q_r = np.linspace(first_q, last_q, num=(num_qs+1))

    # indices of Q rings
    q_inds = np.digitize(q_values, np.array(q_r))
    # discard the indices greater than number of Q rings
    q_inds[q_inds > num_qs] = 0

    # Edge values of each Q rings
    q_ring_val = []

    for i in range(0, num_qs):
        if i < num_qs:
            q_ring_val.append(q_r[i])
            q_ring_val.append(q_r[i + 1])
        else:
            q_ring_val.append(q_r[num_qs-1])

    q_ring_val = np.asarray(q_ring_val)

    return q_ring_val, q_inds


def q_step_val1(q_val, num_qs, first_q, delta_q, *args):
    """
    This will provide q rings edge values when there is a step value
    between rings. Step value can be same or different steps between
    each q ring

    Parameters
    ----------
    q_val : ndarray
        q space values for each pixel in the detector
        shape is ([detector_size[0]*detector_size[1]], ) or
        ([detector_size[0]*detector_size[1]], 1)

    num_qs : int
        number of q rings

    first_q : float
        q value of the first q ring

    delta_q : float
        thickness of the q ring

    *args : tuple
        step value for the next q ring from the end of the previous
        q ring. same step - same step values between q rings (one value)
        different steps - different step value between q rings (provide
        step value for each q ring eg: 6 rings provide 5 step values)

    Returns
    -------
    q_ring_val : ndarray
        edge values of q the required rings

    q_inds : ndarray
        indices of the q values for the required rings

    """
    q_values = _validate_q1(q_val, delta_q)

    q_ring_val = []

    for arg in args:
        if arg < 0:
            raise ValueError("step_q(step value for the next Q ring from the "
                             "end of the previous ring) has to be positive ")

    if len(args) == 1:
        #  when there is a same values of step between q rings
        #  the edge values of q rings will be
        q_ring_val = first_q + np.r_[0, np.cumsum(np.tile([delta_q,
                                                           float(args[0])],
                                                          num_qs))][:-1]
    else:
        # when there is a different step values between each q ring
        #  edge values of the q rings will be
        if len(args) == (num_qs-1):
            q_ring_val.append(first_q)
            for arg in args:
                q_ring_val.append(q_ring_val[-1] + delta_q)
                q_ring_val.append(q_ring_val[-1] + float(arg))
            q_ring_val.append(q_ring_val[-1] + delta_q)
        else:
            raise ValueError("Provide step value for each q ring ")

    # indices of Q rings
    q_inds = np.digitize(q_values, np.array(q_ring_val))

    # to discard every-other bin and set the discarded bins indices to 0
    q_inds[q_inds % 2 == 0] = 0
    # change the indices of odd number of rings
    indx = q_inds > 0
    q_inds[indx] = (q_inds[indx] + 1) // 2

    return q_ring_val, q_inds
