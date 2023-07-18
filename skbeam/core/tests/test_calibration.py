# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# @author: Li Li (lili@bnl.gov)                                        #
# created on 08/19/2014                                                #
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
from __future__ import absolute_import, division, print_function

import numpy as np

import skbeam.core.calibration as calibration
import skbeam.core.calibration as core

from .utils import gauss_gen


def _draw_gaussian_rings(shape, calibrated_center, r_list, r_width):
    R = core.radial_grid(calibrated_center, shape)
    II = np.zeros_like(R)

    for r in r_list:
        tmp = 100 * np.exp(-(((R - r) / r_width) ** 2))
        II += tmp

    return II


def test_refine_center():
    center = np.array((500, 550))
    II = _draw_gaussian_rings((1000, 1001), center, [50, 75, 100, 250, 500], 5)

    nx_opts = [None, 300]
    for nx in nx_opts:
        out = calibration.refine_center(
            II, center + 1, (1, 1), phi_steps=20, nx=nx, min_x=10, max_x=300, window_size=5, thresh=0, max_peaks=4
        )

        assert np.all(np.abs(center - out) < 0.1)


def test_blind_d():
    name = "Si"
    wavelength = 0.18
    window_size = 5
    threshold = 0.1
    cal = calibration.calibration_standards[name]

    tan2theta = np.tan(cal.convert_2theta(wavelength))

    D = 200
    expected_r = D * tan2theta

    bin_centers = np.linspace(0, 50, 2000)
    II = np.zeros_like(bin_centers)
    for r in expected_r:
        II += gauss_gen(bin_centers, r, 100, 0.2)
    d, dstd = calibration.estimate_d_blind(
        name, wavelength, bin_centers, II, window_size, len(expected_r), threshold
    )
    assert np.abs(d - D) < 1e-6
