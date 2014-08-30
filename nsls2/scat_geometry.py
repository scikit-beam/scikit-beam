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
    This module is for finding q values for following
    different  scattering geometries

    (1) saxs (small angle x-ray scattering)
    (2) waxs (wide angle x-ray scattering)
    (3) gisaxs (grazing-incidence small angle x-ray scattering)

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
import logging
logger = logging.getLogger(__name__)


def q_pattern(detector_size, pixel_size,  dist_sample,
              calibrated_center, wavelength, geometry,
              rod_geometry=None, ref_beam=None,
              incident_angle=None):
    """
    This module is for finding the Q values when the
    scattering geometry is saxs, waxs and gisaxs

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

    geometry : str
        scattering geometry (saxs, waxs or gisaxs)

    rod_geometry : str
        geometry of the rod (horizontal or not)

    ref_beam : tuple
        2 element tuple defining (x y) reflected beam (mm)

    incident_angle : float
        incident angle of the beam (degrees)

    Returns
    -------
    q_values : ndarray
        NxN array of Q(reciprocal) space values

    Raises
    ------
    Exception
        Possible causes:
            Raised when the scattering geometry is not specified
    """

    # angular_wave_number
    k = 2*np.pi / wavelength

    if (geometry == 'saxs') | (geometry == 'waxs'):
        # A monochromatic beam of incident wave vector falls
        # on the sample. The scattered intensity is collected
        # as a function of the scattering angle 2θ.
        # waxs is the same technique as saxs  only the distance
        # from sample to the detector is shorter and thus
        # diffraction maxima at larger angles are observed.

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

    elif (geometry == 'gisaxs'):
        # A monochromatic x-ray beam with the wave vector ki is directed
        # on a surface with a very small incident angle αi with respect
        # to the surface. The x-rays are scattered along kf in the
        # direction (2θ, αf). The Cartesian z-axis is the normal to the
        # surface plane,the x-axis is the direction along the surface
        # parallel to the beam and the y-axis perpendicular to it.

         # incident angle αi
        in_angle_i = (incident_angle)/180.0*np.pi

        x_origin = (calibrated_center[0] + ref_beam[0])/2
        y_origin = (calibrated_center[1] + ref_beam[1])/2

        x_pix = np.reshape(np.arange(detector_size[0]) -
                           x_origin, (1, -1))
        y_pix = np.reshape(np.arange(detector_size[1]) -
                           y_origin, (-1, 1))

        x_mm = x_pix * pixel_size[0]
        y_mm = y_pix * pixel_size[1]

        if (rod_geometry == 'horizontel'):
            # angle αf
            in_angle_f = np.arctan(x_mm/dist_sample)

            two_theta = np.arctan(y_mm/dist_sample)
            # x component
            q_x = k*(-np.cos(in_angle_f) * np.cos(two_theta) +
                     np.cos(in_angle_i))
        else:
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

    else:
        raise Exception(" ---- Specify the scattering geometry --- ")

    return q_values
