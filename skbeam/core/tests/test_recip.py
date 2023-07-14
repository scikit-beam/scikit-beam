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
from __future__ import absolute_import, division, print_function

import os

import numpy as np
import numpy.testing as npt
import pytest
from numpy.testing import assert_array_almost_equal

from skbeam.core import recip


@pytest.mark.skipif(os.name == "nt", reason="Test is not supported on Windows")
def test_process_to_q():
    detector_size = (256, 256)
    pixel_size = (0.0135 * 8, 0.0135 * 8)
    calibrated_center = (256 / 2.0, 256 / 2.0)
    dist_sample = 355.0

    energy = 640  # (  in eV)
    # HC_OVER_E to convert from Energy to wavelength (Lambda)
    hc_over_e = 12398.4
    wavelength = hc_over_e / energy  # (Angstrom )

    ub_mat = np.array(
        [
            [-0.01231028454, 0.7405370482, 0.06323870032],
            [0.4450897473, 0.04166852402, -0.9509449389],
            [-0.7449130975, 0.01265920962, -0.5692399963],
        ]
    )

    setting_angles = np.array([[40.0, 15.0, 30.0, 25.0, 10.0, 5.0], [90.0, 60.0, 0.0, 30.0, 10.0, 5.0]])
    # delta=40, theta=15, chi = 90, phi = 30, mu = 10.0, gamma=5.0
    pdict = {}
    pdict["setting_angles"] = setting_angles
    pdict["detector_size"] = detector_size
    pdict["pixel_size"] = pixel_size
    pdict["calibrated_center"] = calibrated_center
    pdict["dist_sample"] = dist_sample
    pdict["wavelength"] = wavelength
    pdict["ub"] = ub_mat
    # ensure invalid entries for frame_mode actually fail

    # todo test frame_modes 1, 2, and 3
    # test that the values are coming back as expected for frame_mode=4
    hkl = recip.process_to_q(**pdict)

    # Known HKL values for the given six angles)
    # each entry in list is (pixel_number, known hkl value)
    known_hkl = [
        (32896, np.array([-0.15471196, 0.19673939, -0.11440936])),
        (98432, np.array([0.10205953, 0.45624416, -0.27200778])),
    ]

    for pixel, kn_hkl in known_hkl:
        npt.assert_array_almost_equal(hkl[pixel], kn_hkl, decimal=8)

    # smoketest the frame_mode variable
    pass_list = recip.process_to_q.frame_mode
    pass_list.append(None)
    for passes in pass_list:
        recip.process_to_q(frame_mode=passes, **pdict)


def _process_to_q_exception(param_dict, frame_mode):
    with pytest.raises(KeyError):
        recip.process_to_q(frame_mode=frame_mode, **param_dict)


@pytest.mark.parametrize("fails", [0, 5, "cat"])
@pytest.mark.skipif(os.name == "nt", reason="Test is not supported on Windows")
def test_frame_mode_fail(fails):
    detector_size = (256, 256)
    pixel_size = (0.0135 * 8, 0.0135 * 8)
    calibrated_center = (256 / 2.0, 256 / 2.0)
    dist_sample = 355.0

    energy = 640  # (  in eV)
    # HC_OVER_E to convert from Energy to wavelength (Lambda)
    hc_over_e = 12398.4
    wavelength = hc_over_e / energy  # (Angstrom )

    ub_mat = np.array(
        [
            [-0.01231028454, 0.7405370482, 0.06323870032],
            [0.4450897473, 0.04166852402, -0.9509449389],
            [-0.7449130975, 0.01265920962, -0.5692399963],
        ]
    )

    setting_angles = np.array([[40.0, 15.0, 30.0, 25.0, 10.0, 5.0], [90.0, 60.0, 0.0, 30.0, 10.0, 5.0]])
    # delta=40, theta=15, chi = 90, phi = 30, mu = 10.0, gamma=5.0
    pdict = {}
    pdict["setting_angles"] = setting_angles
    pdict["detector_size"] = detector_size
    pdict["pixel_size"] = pixel_size
    pdict["calibrated_center"] = calibrated_center
    pdict["dist_sample"] = dist_sample
    pdict["wavelength"] = wavelength
    pdict["ub"] = ub_mat

    _process_to_q_exception(pdict, fails)


def test_hkl_to_q():
    b = np.array([[-4, -3, -2], [-1, 0, 1], [2, 3, 4], [6, 9, 10]])

    b_norm = np.array([5.38516481, 1.41421356, 5.38516481, 14.73091986])

    npt.assert_array_almost_equal(b_norm, recip.hkl_to_q(b))


def test_gisaxs():
    incident_beam = (1.0, 1.0)
    reflected_beam = (3.0, 3.0)
    pixel_size = (1.0, 1.0)
    detector_size = (5, 4)
    dist_sample = 5.0
    wavelength = 2 * np.pi * 0.01
    theta_i = 0.0

    g_output = recip.gisaxs(
        incident_beam, reflected_beam, pixel_size, detector_size, dist_sample, wavelength, theta_i=theta_i
    )

    theta_f_target = 10 ** (-7) * np.array([-1.0, 0.0, 1.0, 2.0])
    alpha_f_target = 10 ** (-7) * np.array([-4.0, -2.0, 7.99387344e-14, 2.0, 4.0])

    assert_array_almost_equal(0.78539816, g_output.tilt_angle, decimal=8)
    assert_array_almost_equal(2 * 10 ** (-7), g_output.alpha_i, decimal=8)
    assert_array_almost_equal(theta_f_target, g_output.theta_f[1, :], decimal=8)
    assert_array_almost_equal(alpha_f_target, g_output.alpha_f[:, 1])
