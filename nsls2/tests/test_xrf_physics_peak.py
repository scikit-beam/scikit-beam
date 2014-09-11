# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# @author: Li Li (lili@bnl.gov)                                        #
# created on 07/16/2014                                                #
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


from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
from numpy.testing import (assert_allclose, assert_array_almost_equal)

from nsls2.fitting.model.physics_peak import (gauss_peak, gauss_step, gauss_tail,
                                              elastic_peak, compton_peak)


def test_gauss_peak():
    """
    test of gauss function from xrf fit
    """
    area = 1
    cen  = 0
    std = 1
    x = np.arange(-3, 3, 0.5)
    out = gauss_peak(x, area, cen, std)

    y_true = [0.00443185, 0.0175283, 0.05399097, 0.1295176, 0.24197072, 0.35206533,
              0.39894228, 0.35206533, 0.24197072, 0.1295176, 0.05399097, 0.0175283]

    assert_array_almost_equal(y_true, out)

    return


def test_gauss_step():
    """
    test of gaussian step function from xrf fit
    """

    y_true = [1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
              1.00000000e+00,   9.99999999e-01,   9.99999713e-01,
              9.99968329e-01,   9.98650102e-01,   9.77249868e-01,
              8.41344746e-01,   5.00000000e-01,   1.58655254e-01,
              2.27501319e-02,   1.34989803e-03,   3.16712418e-05]
    area = 1
    cen = 0
    std = 1
    x = np.arange(-10, 5, 1)
    peak_e = 1.0
    out = gauss_step(x, area, cen, std, peak_e)

    assert_array_almost_equal(y_true, out)
    return


def test_gauss_tail():
    """
    test of gaussian tail function from xrf fit
    """

    y_true = [7.48518299e-05, 2.03468369e-04, 5.53084370e-04, 1.50343919e-03,
              4.08677027e-03, 1.11086447e-02, 3.01566200e-02, 8.02175541e-02,
              1.87729388e-01, 3.03265330e-01, 2.61578292e-01, 3.75086265e-02,
              2.22560560e-03, 5.22170501e-05, 4.72608544e-07]

    area = 1
    cen = 0
    std = 1
    x = np.arange(-10, 5, 1)
    gamma = 1.0
    out = gauss_tail(x, area, cen, std, gamma)

    assert_array_almost_equal(y_true, out)

    return


def test_elastic_peak():
    """
    test of elastic peak from xrf fit
    """

    y_true = [0.00085311,  0.00164853,  0.00307974,  0.00556237,  0.00971259,
              0.01639604,  0.02675911,  0.04222145,  0.06440556,  0.09498223,
              0.13542228,  0.18666663,  0.24875512,  0.32048386,  0.39918028,
              0.48068522,  0.55960456,  0.62984039,  0.68534389,  0.72096698,
              0.73324816,  0.72096698,  0.68534389,  0.62984039,  0.55960456,
              0.48068522,  0.39918028,  0.32048386,  0.24875512,  0.18666663,
              0.13542228,  0.09498223,  0.06440556,  0.04222145,  0.02675911,
              0.01639604,  0.00971259,  0.00556237,  0.00307974,  0.00164853]

    area = 1
    energy = 10
    offset = 0.01
    fanoprime = 0.01

    ev = np.arange(8, 12, 0.1)
    out, sigma = elastic_peak(ev, energy, offset,
                              fanoprime, area)

    assert_array_almost_equal(y_true, out)
    return


def test_compton_peak():
    """
    test of compton peak from xrf fit
    """

    y_true  = [0.13322374,  0.15369844,  0.18701130,  0.24010139,  0.32232808,
               0.44551425,  0.62348701,  0.87091681,  1.20134347,  1.62445241,
               2.14291102,  2.74933771,  3.42416929,  4.13521971,  4.83951630,
               5.48755599,  6.02952905,  6.42247263,  6.63693264,  6.57925536,
               6.30502092,  5.84781459,  5.25108917,  4.56740794,  3.85083566,
               3.15005570,  2.50337782,  1.93622014,  1.46102640,  1.07908755,
               0.78347806,  0.56230190,  0.40161350,  0.28763835,  0.20817573,
               0.15326083,  0.11527037,  0.08868334,  0.06968182,  0.05572342]

    energy = 10
    offset = 0.01
    fano = 0.01
    angle = 90
    fwhm_corr = 1
    amp = 1
    f_step = 0
    f_tail = 0.1
    gamma = 10
    hi_f_tail = 0.1
    hi_gamma = 1
    ev = np.arange(8, 12, 0.1)

    out, sigma, factor = compton_peak(ev, energy, offset, fano, angle,
                                      fwhm_corr, amp, f_step, f_tail,
                                      gamma, hi_f_tail, hi_gamma)

    assert_array_almost_equal(y_true, out)
    return
