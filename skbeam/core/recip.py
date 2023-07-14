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
from __future__ import absolute_import, division, print_function

import time
from collections import namedtuple

import numpy as np

from .utils import verbosedict

try:
    from pyFAI import geometry as geo
except ImportError:
    geo = None


import logging

logger = logging.getLogger(__name__)


def process_to_q(
    setting_angles, detector_size, pixel_size, calibrated_center, dist_sample, wavelength, ub, frame_mode=None
):
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
    try:
        from ..ext import ctrans
    except ImportError:
        raise NotImplementedError(
            "ctrans is not available on your platform. See"
            "https://github.com/scikit-beam/scikit-beam/issues/418"
            "to follow updates to this problem."
        )

    # Set default threads

    # set default frame_mode
    if frame_mode is None:
        frame_mode = 4
    else:
        str_to_int = verbosedict((k, j + 1) for j, k in enumerate(process_to_q.frame_mode))
        frame_mode = str_to_int[frame_mode]
    # ensure the ub matrix is an array
    ub = np.asarray(ub)
    # ensure setting angles is a 2-D
    setting_angles = np.atleast_2d(setting_angles)
    if setting_angles.ndim != 2:
        raise ValueError(
            "setting_angles is expected to be a 2-D array with"
            " dimensions [num_images][num_angles]. You provided "
            "an array with dimensions {0}"
            "".format(setting_angles.shape)
        )
    if setting_angles.shape[1] != 6:
        raise ValueError(
            "It is expected that there should be six angles in "
            "the setting_angles parameter. You provided {0}"
            " angles.".format(setting_angles.shape[1])
        )
    # *********** Converting to Q   **************

    # starting time for the process
    t1 = time.time()

    # ctrans - c routines for fast data analysis
    hkl = ctrans.ccdToQ(
        angles=setting_angles * np.pi / 180.0,
        mode=frame_mode,
        ccd_size=(detector_size),
        ccd_pixsize=(pixel_size),
        ccd_cen=(calibrated_center),
        dist=dist_sample,
        wavelength=wavelength,
        UBinv=np.linalg.inv(ub),
    )

    # ending time for the process
    t2 = time.time()
    logger.info(
        "Processing time for {0} {1} x {2} images took {3} seconds."
        "".format(setting_angles.shape[0], detector_size[0], detector_size[1], (t2 - t1))
    )
    return hkl


# Assign frame_mode as an attribute to the process_to_q function so that the
# autowrapping knows what the valid options are
process_to_q.frame_mode = ["theta", "phi", "cart", "hkl"]


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


def calibrated_pixels_to_q(detector_size, pyfai_kwargs):
    """
    For a given detector and pyfai calibrated geometry give back the q value
    for each pixel in the detector.

    Parameters
    ----------
    detector_size : tuple
        2 element tuple defining the number of pixels in the detector. Order is
        (num_columns, num_rows)
    pyfai_kwargs: dict
        The dictionary of pyfai geometry kwargs, given by pyFAI's calibration
        Ex: dist, poni1, poni2, rot1, rot2, rot3, splineFile, wavelength,
        detector, pixel1, pixel2

    Returns
    -------
    q_val : ndarray
        Reciprocal values for each pixel shape is [num_rows * num_columns]
    """
    if geo is None:
        raise RuntimeError("You must have pyFAI installed to use this " "function.")
    a = geo.Geometry(**pyfai_kwargs)
    return a.qArray(detector_size)


gisaxs_output = namedtuple(
    "gisaxs_output", ["alpha_i", "theta_f", "alpha_f", "tilt_angle", "qx", "qy", "qz", "qr"]
)


def gisaxs(incident_beam, reflected_beam, pixel_size, detector_size, dist_sample, wavelength, theta_i=0.0):
    """
    This function will provide scattering wave vector(q) components(x, y, z),
    q parallel and incident and reflected angles for grazing-incidence small
    angle X-ray scattering (GISAXS) geometry.

    Parameters
    ----------
    incident_beam : tuple
        x and y co-ordinates of the incident beam in pixels
    reflected_beam : tuple
        x and y co-ordinates of the reflected beam in pixels
    pixel_size : tuple
        pixel_size in um
    detector_size: tuple
        2 element tuple defining no. of pixels(size) in the
        detector X and Y direction
    dist_sample : float
       sample to detector distance, in meters
    wavelength : float
        wavelength of the x-ray beam in Angstroms
    theta_i : float, optional
        out of plane angle, default 0.0

    Returns
    -------
    namedtuple
        `gisaxs_output` object is returned
        This `gisaxs_output` object contains, in this order:

        - alpha_i : float
          incident angle
        - theta_f : array
          out of plane angle
          shape (detector_size[0], detector_size[1])
        - alpha_f : array
          exit angle
          shape (detector_size[0], detector_size[1])
        - tilt_angle : float
          tilt angle
        - qx : array
          x component of the scattering wave vector
          shape (detector_size[0], detector_size[1])
        - qy : array
          y component of the scattering wave vector
          shape (detector_size[0], detector_size[1])
        - qz : array
          z component of the scattering wave vector
          shape (detector_size[0], detector_size[1])
        - qr : array
          q parallel component
          shape (detector_size[0], detector_size[1])

    Notes
    -----
    This implementation is based on published work. [1]_

    References
    ----------
    .. [1] R. Lazzari, "IsGISAXS: a program for grazing-incidence small-
       angle X-ray scattering analysis of supported islands," J. Appl.
       Cryst., vol 35, p 406-421, 2002.
    """
    inc_x, inc_y = incident_beam
    refl_x, refl_y = reflected_beam

    # convert pixel_size to meters
    pixel_size = np.asarray(pixel_size) * 10 ** (-6)

    # tilt angle
    tilt_angle = np.arctan2((refl_x - inc_x) * pixel_size[0], (refl_y - inc_y) * pixel_size[1])
    # incident angle
    alpha_i = np.arctan2((refl_y - inc_y) * pixel_size[1], dist_sample) / 2.0

    y, x = np.indices(detector_size)
    # exit angle
    alpha_f = np.arctan2((y - inc_y) * pixel_size[1], dist_sample) - alpha_i
    # out of plane angle
    two_theta = np.arctan2((x - inc_x) * pixel_size[0], dist_sample)
    theta_f = two_theta / 2 - theta_i
    # wave number
    wave_number = 2 * np.pi / wavelength

    # x component
    qx = (np.cos(alpha_f) * np.cos(2 * theta_f) - np.cos(alpha_i) * np.cos(2 * theta_i)) * wave_number

    # y component
    # the variables post-fixed with an underscore are intermediate steps
    qy_ = np.cos(alpha_f) * np.sin(2 * theta_f) - np.cos(alpha_i) * np.sin(2 * theta_i)
    qz_ = np.sin(alpha_f) + np.sin(alpha_i)
    qy = (qz_ * np.sin(tilt_angle) + qy_ * np.cos(tilt_angle)) * wave_number

    # z component
    qz = (qz_ * np.cos(tilt_angle) - qy_ * np.sin(tilt_angle)) * wave_number

    # q parallel
    qr = np.sqrt(qx**2 + qy**2)

    return gisaxs_output(alpha_i, theta_f, alpha_f, tilt_angle, qx, qy, qz, qr)
