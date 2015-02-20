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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
from nose.tools import raises

from skxray.testing.decorators import known_fail_if
import numpy.testing as npt
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)

from nose.tools import assert_equal, assert_true, raises
from skxray import recip


@known_fail_if(six.PY3)
def test_process_to_q():
    detector_size = (256, 256)
    pixel_size = (0.0135*8, 0.0135*8)
    calibrated_center = (256/2.0, 256/2.0)
    dist_sample = 355.0

    energy = 640  # (  in eV)
    # HC_OVER_E to convert from Energy to wavelength (Lambda)
    hc_over_e = 12398.4
    wavelength = hc_over_e / energy  # (Angstrom )

    ub_mat = np.array([[-0.01231028454, 0.7405370482, 0.06323870032],
                       [0.4450897473, 0.04166852402, -0.9509449389],
                       [-0.7449130975, 0.01265920962, -0.5692399963]])

    setting_angles = np.array([[40., 15., 30., 25., 10., 5.],
                              [90., 60., 0., 30., 10., 5.]])
    # delta=40, theta=15, chi = 90, phi = 30, mu = 10.0, gamma=5.0
    pdict = {}
    pdict['setting_angles'] = setting_angles
    pdict['detector_size'] = detector_size
    pdict['pixel_size'] = pixel_size
    pdict['calibrated_center'] = calibrated_center
    pdict['dist_sample'] = dist_sample
    pdict['wavelength'] = wavelength
    pdict['ub'] = ub_mat
    # ensure invalid entries for frame_mode actually fail

    # todo test frame_modes 1, 2, and 3
    # test that the values are coming back as expected for frame_mode=4
    hkl = recip.process_to_q(**pdict)

    # Known HKL values for the given six angles)
    # each entry in list is (pixel_number, known hkl value)
    known_hkl = [(32896, np.array([-0.15471196, 0.19673939, -0.11440936])),
                 (98432, np.array([0.10205953,  0.45624416, -0.27200778]))]

    for pixel, kn_hkl in known_hkl:
        npt.assert_array_almost_equal(hkl[pixel], kn_hkl, decimal=8)

    # smoketest the frame_mode variable
    pass_list = recip.process_to_q.frame_mode
    pass_list.append(None)
    for passes in pass_list:
        recip.process_to_q(frame_mode=passes, **pdict)


@raises(KeyError)
def _process_to_q_exception(param_dict, frame_mode):
    recip.process_to_q(frame_mode=frame_mode, **param_dict)


def test_frame_mode_fail():
    detector_size = (256, 256)
    pixel_size = (0.0135*8, 0.0135*8)
    calibrated_center = (256/2.0, 256/2.0)
    dist_sample = 355.0

    energy = 640  # (  in eV)
    # HC_OVER_E to convert from Energy to wavelength (Lambda)
    hc_over_e = 12398.4
    wavelength = hc_over_e / energy  # (Angstrom )

    ub_mat = np.array([[-0.01231028454, 0.7405370482, 0.06323870032],
                       [0.4450897473, 0.04166852402, -0.9509449389],
                       [-0.7449130975, 0.01265920962, -0.5692399963]])

    setting_angles = np.array([[40., 15., 30., 25., 10., 5.],
                              [90., 60., 0., 30., 10., 5.]])
    # delta=40, theta=15, chi = 90, phi = 30, mu = 10.0, gamma=5.0
    pdict = {}
    pdict['setting_angles'] = setting_angles
    pdict['detector_size'] = detector_size
    pdict['pixel_size'] = pixel_size
    pdict['calibrated_center'] = calibrated_center
    pdict['dist_sample'] = dist_sample
    pdict['wavelength'] = wavelength
    pdict['ub'] = ub_mat

    for fails in [0, 5, 'cat']:
        yield _process_to_q_exception, pdict, fails


def test_hkl_to_q():
    b = np.array([[-4, -3, -2],
                  [-1, 0, 1],
                  [2, 3, 4],
                  [6, 9, 10]])

    b_norm = np.array([5.38516481, 1.41421356, 5.38516481,
                       14.73091986])

    npt.assert_array_almost_equal(b_norm, recip.hkl_to_q(b))


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
