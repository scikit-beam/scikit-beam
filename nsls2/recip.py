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
    setting_angles : ndarray
        six angles of all the images - Nx6 array
        delta, theta, chi, phi, mu, gamma (degrees)

    detector_size : tuple
        2 element tuple defining no. of pixels(size) in the
        detector X and Y direction(mm)

    pixel_size : tuple
        2 element tuple defining the (x y) dimensions of the
        pixel (mm)

    calibrated_center : tuple
        2 element tuple defining the (x y) center of the
        detector (mm)

    dist_sample : float
        distance from the sample to the detector (mm)

    wavelength : float
        wavelength of incident radiation (Angstroms)

    ub_mat :  ndarray
        UB matrix (orientation matrix) 3x3 matrix

    Returns
    -------
    tot_set : ndarray
        (Qx, Qy, Qz) - HKL values - Nx3 matrix

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

    setting_angles = np.atleast_2d(setting_angles)
    setting_angles.shape
    if setting_angles.ndim != 2:
        raise ValueError()
    if setting_angles.shape[1] != 6:
        raise ValueError()

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

    Parameters
    ---------
    tot_set : ndarray
        (Qx, Qy, Qz) - HKL values - Nx3 array

    istack : ndarray
        intensity array of the images - Nx1 array

    q_min : ndarray, optional
        minimum values of the voxel [Qx, Qy, Qz]_min

    q_max : ndarray, optional
        maximum values of the voxel [Qx, Qy, Qz]_max

    dqn : ndarray, optional
        No. of grid parts (bins) [Nqx, Nqy, Nqz]

    Returns
    -------
    grid_mean : ndarray
        intensity grid.  The values in this grid are the
        mean of the values that fill with in the grid.

    grid_error : ndarray
        This is the standard error of the value in the
        grid box.

    grid_occu : ndarray
        The number of data points that fell in the grid.

    n_out_of_bounds : int
        No. of data points that were outside of the gridded region.

    Raises
    ------
    ValueError
        Possible causes:
            Raised when the HKL values are not provided
    """

    tot_set = np.atleast_2d(tot_set)
    tot_set.shape
    if tot_set.ndim != 2:
        raise ValueError(
            "The tot_set.nidm must be 2, not {}".format(tot_set.ndim))
    if tot_set.shape[1] != 3:
        raise ValueError(
            "The shape of tot_set must be Nx3 not "
            "{}X{}".format(*tot_set.shape))

    # prepare min, max,... from defaults if not set
    if q_min is None:
        q_min = np.min(tot_set, axis=0)
    if q_max is None:
        q_max = np.max(tot_set, axis=0)
    if dqn is None:
        dqn = [100, 100, 100]

    # creating (Qx, Qy, Qz, I) Nx4 array - HKL values and Intensity
    # getting the intensity value for each pixel
    tot_set = np.insert(tot_set, 3, np.ravel(i_stack), axis=1)

    #            3D grid of the data set
    #             *** Gridding Data ****

    # starting time for gridding
    t1 = time.time()

    # ctrans - c routines for fast data analysis

    (grid_mean, grid_occu,
         grid_error, n_out_of_bounds) = ctrans.grid3d(tot_set,
                                                      q_min, q_max, dqn,
                                                      norm=1)

    # ending time for the gridding
    t2 = time.time()
    logger.info("Done processed in %f seconds", (t2-t1))

    # No. of values zero in the grid
    empt_nb = (grid_occu == 0).sum()

    if n_out_of_bounds:
        logger.debug("There are %.2e points outside the grid ",
                     n_out_of_bounds)
    logger.debug("There are %2e bins in the grid ", grid_mean.size)
    if empt_nb:
        logger.debug("There are %.2e values zero in the grid ", empt_nb)

    return grid_mean, grid_occu, grid_error, n_out_of_bounds


def convert_to_q_saxs(detector_size, pixel_size, dist_sample,
                      calibrated_center, wavelength):
    """
    This module is for finding Q values for saxs (small
    angle scattering geometry) scattering geometries

    A monochromatic beam of incident wave vector falls
    on the sample. The scattered intensity is collected
    as a function of the scattering angle (2theta).

    Parameters
    ----------
    detector_size : tuple
        2 element tuple defining no. of pixels(size) in the
        detector X and Y direction(mm)

    pixel_size : tuple
        2 element tuple defining the (x y) dimensions of the
        pixel (mm)

    dist_sample : float
        distance from the sample to the detector (mm)

    calibrated_center : tuple
        2 element tuple defining the (x y) center of the
        detector (mm)

    wavelength : float
        wavelength of incident radiation (Angstroms)

    Returns
    -------
    q_values : ndarray
        NxN array of Q(reciprocal) space values

    """

    # angular_wave_number
    k = 2*np.pi / wavelength

    x_pix = np.reshape(np.arange(detector_size[0]) -
                           calibrated_center[0], (1, -1))
    y_pix = np.reshape(np.arange(detector_size[1]) -
                           calibrated_center[1], (-1, 1))

    x_mm = x_pix * pixel_size[0]
    y_mm = y_pix * pixel_size[1]

    pix_distance = np.sqrt(x_mm**2 + y_mm**2)

    # scattering angle
    two_theta = np.arctan(pix_distance/dist_sample)
    # q values
    q_values = 2 * k * np.sin(two_theta/2)

    return q_values


def convert_to_q_waxs(detector_size, pixel_size,  dist_sample,
                      calibrated_center, wavelength):
    """
    This module is for finding Q values for waxs (wide angle
    scattering geometry) scattering geometries

    A monochromatic beam of incident wave vector falls
    on the sample. The scattered intensity is collected
    as a function of the scattering angle (2theta).
    waxs is the same technique as saxs only the distance
    from sample to the detector is shorter and thus
    diffraction maxima at larger angles are observed.

    Parameters
    ----------
    detector_size : tuple
        2 element tuple defining no. of pixels(size) in the
        detector X and Y direction(mm)

    pixel_size : tuple
        2 element tuple defining the (x y) dimensions of the
        pixel (mm)

    dist_sample : float
        distance from the sample to the detector (mm)

    calibrated_center : tuple
        2 element tuple defining the (x y) center of the
        detector (mm)

    wavelength : float
        wavelength of incident radiation (Angstroms)

    Returns
    -------
    q_values : ndarray
        NxN array of Q(reciprocal) space values

    """

    # angular_wave_number
    k = 2*np.pi / wavelength

    x_pix = np.reshape(np.arange(detector_size[0]) -
                           calibrated_center[0], (1, -1))
    y_pix = np.reshape(np.arange(detector_size[1]) -
                           calibrated_center[1], (-1, 1))

    x_mm = x_pix * pixel_size[0]
    y_mm = y_pix * pixel_size[1]

    pix_distance = np.sqrt(x_mm**2 + y_mm**2)

    # scattering angle
    two_theta = np.arctan(pix_distance/dist_sample)
    # q values
    q_values = 2 * k * np.sin(two_theta/2)

    return q_values


def convert_to_q_giaxs(detector_size, pixel_size,  dist_sample,
              calibrated_center, wavelength, ref_beam,
              incident_angle, rod_geometry=None):
    """
    This module is for finding Q values for gisaxs
    (grazing-incidence small angle x-ray scattering)
    scattering geometry

    A monochromatic x-ray beam with the wave vector ki is
    directed on a surface with a very small incident angle
    alpha(i) with respect to the surface. The x-rays are
    scattered along kf in the direction (2theta, alpha(f)).
    The Cartesian z-axis is the normal to the surface plane,
    the x-axis is the direction along the surface parallel
    to the beam and the y-axis perpendicular to it.

    Parameters
    ----------
    detector_size : tuple
        2 element tuple defining no. of pixels(size) in the
        detector X and Y direction(mm)

    pixel_size : tuple
        2 element tuple defining the (x y) dimensions of the
        pixel (mm)

    dist_sample : float
        distance from the sample to the detector (mm)

    calibrated_center : tuple
        2 element tuple defining the (x y) center of the
        detector (mm)

    wavelength : float
        wavelength of incident radiation (Angstroms)

    ref_beam : tuple
        2 element tuple defining (x y) reflected beam (mm)

    incident_angle : float
        incident angle of the beam (degrees)

    rod_geometry : str, optional
        geometry of the rod (horizontal or not)

    Returns
    -------
    q_values : ndarray
        NxN array of Q(reciprocal) space values

    """

    # angular_wave_number
    k = 2*np.pi / wavelength

    # incident angle alpha(i)
    in_angle_i = (incident_angle)/180.0*np.pi

    x_origin = (calibrated_center[0] + ref_beam[0])/2
    y_origin = (calibrated_center[1] + ref_beam[1])/2

    x_pix = np.reshape(np.arange(detector_size[0]) -
                           x_origin, (1, -1))
    y_pix = np.reshape(np.arange(detector_size[1]) -
                           y_origin, (-1, 1))

    x_mm = x_pix * pixel_size[0]
    y_mm = y_pix * pixel_size[1]

    if (rod_geometry == 'horizontal'):
        # angle alpha(f)
        in_angle_f = np.arctan(x_mm/dist_sample)

        two_theta = np.arctan(y_mm/dist_sample)
        # x component
        q_x = k*(-np.cos(in_angle_f) * np.cos(two_theta) +
                     np.cos(in_angle_i))
    else:
        # angle alpha(f)
        in_angle_f = np.arctan(y_mm/dist_sample)

        two_theta = np.arctan(x_mm/dist_sample)
        # x component
        q_x = k*(-np.cos(in_angle_f) * np.cos(two_theta) +
                     np.cos(in_angle_i))

        # y component
        q_y = k*np.cos(in_angle_f) * np.sin(two_theta)

        # z component
        q_z = np.resize(k*np.sin(in_angle_f) + k*np.sin(in_angle_i),
                        (detector_size[1], detector_size[0]))

        q_values = np.sqrt(q_x ** 2 + q_y ** 2 + q_z ** 2)

    return q_values
