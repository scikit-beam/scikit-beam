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
from __future__ import absolute_import, division, print_function

import numpy as np
from numpy.testing import assert_array_almost_equal

from skbeam.core.fitting import (
    ComptonModel,
    ElasticModel,
    compton,
    elastic,
    gamma_dist,
    gaussian,
    gaussian_tail,
    gausssian_step,
    lorentzian,
    lorentzian2,
    nbinom_dist,
    poisson_dist,
    pvoigt,
    voigt,
)


def test_gauss_peak():
    """
    test of gauss function from xrf fit
    """
    area = 1
    cen = 0
    std = 1
    x = np.arange(-3, 3, 0.5)
    out = gaussian(x, area, cen, std)

    y_true = [
        0.00443185,
        0.0175283,
        0.05399097,
        0.1295176,
        0.24197072,
        0.35206533,
        0.39894228,
        0.35206533,
        0.24197072,
        0.1295176,
        0.05399097,
        0.0175283,
    ]

    assert_array_almost_equal(y_true, out)


def test_gauss_step():
    """
    test of gaussian step function from xrf fit
    """

    y_true = [
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        9.99999999e-01,
        9.99999713e-01,
        9.99968329e-01,
        9.98650102e-01,
        9.77249868e-01,
        8.41344746e-01,
        5.00000000e-01,
        1.58655254e-01,
        2.27501319e-02,
        1.34989803e-03,
        3.16712418e-05,
    ]
    area = 1
    cen = 0
    std = 1
    x = np.arange(-10, 5, 1)
    peak_e = 1.0
    out = gausssian_step(x, area, cen, std, peak_e)

    assert_array_almost_equal(y_true, out)


def test_gauss_tail():
    """
    test of gaussian tail function from xrf fit
    """

    y_true = [
        7.48518299e-05,
        2.03468369e-04,
        5.53084370e-04,
        1.50343919e-03,
        4.08677027e-03,
        1.11086447e-02,
        3.01566200e-02,
        8.02175541e-02,
        1.87729388e-01,
        3.03265330e-01,
        2.61578292e-01,
        3.75086265e-02,
        2.22560560e-03,
        5.22170501e-05,
        4.72608544e-07,
    ]

    area = 1
    cen = 0
    std = 1
    x = np.arange(-10, 5, 1)
    gamma = 1.0
    out = gaussian_tail(x, area, cen, std, gamma)

    assert_array_almost_equal(y_true, out)


def test_elastic_peak():
    """
    test of elastic peak from xrf fit
    """

    y_true = [
        0.00085311,
        0.00164853,
        0.00307974,
        0.00556237,
        0.00971259,
        0.01639604,
        0.02675911,
        0.04222145,
        0.06440556,
        0.09498223,
        0.13542228,
        0.18666663,
        0.24875512,
        0.32048386,
        0.39918028,
        0.48068522,
        0.55960456,
        0.62984039,
        0.68534389,
        0.72096698,
        0.73324816,
        0.72096698,
        0.68534389,
        0.62984039,
        0.55960456,
        0.48068522,
        0.39918028,
        0.32048386,
        0.24875512,
        0.18666663,
        0.13542228,
        0.09498223,
        0.06440556,
        0.04222145,
        0.02675911,
        0.01639604,
        0.00971259,
        0.00556237,
        0.00307974,
        0.00164853,
    ]

    area = 1
    energy = 10
    offset = 0.01
    fanoprime = 0.01
    e_offset = 0
    e_linear = 1
    e_quadratic = 0

    ev = np.arange(8, 12, 0.1)
    out = elastic(ev, area, energy, offset, fanoprime, e_offset, e_linear, e_quadratic)

    assert_array_almost_equal(y_true, out)


def test_compton_peak():
    """
    test of compton peak from xrf fit
    """
    y_true = [
        0.01332237,
        0.01536984,
        0.01870113,
        0.02401014,
        0.03223281,
        0.04455143,
        0.0623487,
        0.08709168,
        0.12013435,
        0.16244524,
        0.2142911,
        0.27493377,
        0.34241693,
        0.41352197,
        0.48395163,
        0.5487556,
        0.6029529,
        0.64224726,
        0.66369326,
        0.65792554,
        0.63050209,
        0.58478146,
        0.52510892,
        0.45674079,
        0.38508357,
        0.31500557,
        0.25033778,
        0.19362201,
        0.14610264,
        0.10790876,
        0.07834781,
        0.05623019,
        0.04016135,
        0.02876383,
        0.02081757,
        0.01532608,
        0.01152704,
        0.00886833,
        0.00696818,
        0.00557234,
    ]

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

    e_offset = 0
    e_linear = 1
    e_quadratic = 0

    ev = np.arange(8, 12, 0.1)

    out = compton(
        ev,
        amp,
        energy,
        offset,
        fano,
        e_offset,
        e_linear,
        e_quadratic,
        angle,
        fwhm_corr,
        f_step,
        f_tail,
        gamma,
        hi_f_tail,
        hi_gamma,
    )

    assert_array_almost_equal(y_true, out)


def test_lorentzian_peak():
    y_true = [
        0.03151583,
        0.03881828,
        0.04897075,
        0.06366198,
        0.0860297,
        0.12242688,
        0.18724111,
        0.31830989,
        0.63661977,
        1.59154943,
        3.18309886,
        1.59154943,
        0.63661977,
        0.31830989,
        0.18724111,
        0.12242688,
        0.0860297,
        0.06366198,
        0.04897075,
        0.03881828,
    ]

    x = np.arange(-1, 1, 0.1)
    a = 1
    cen = 0
    std = 0.1
    out = lorentzian(x, a, cen, std)

    assert_array_almost_equal(y_true, out)


def test_lorentzian_squared_peak():
    y_true = [
        3.12037924e-04,
        4.73393644e-04,
        7.53396180e-04,
        1.27323954e-03,
        2.32512700e-03,
        4.70872613e-03,
        1.10141829e-02,
        3.18309886e-02,
        1.27323954e-01,
        7.95774715e-01,
        3.18309886e00,
        7.95774715e-01,
        1.27323954e-01,
        3.18309886e-02,
        1.10141829e-02,
        4.70872613e-03,
        2.32512700e-03,
        1.27323954e-03,
        7.53396180e-04,
        4.73393644e-04,
    ]

    x = np.arange(-1, 1, 0.1)
    a = 1
    cen = 0
    std = 0.1
    out = lorentzian2(x, a, cen, std)

    assert_array_almost_equal(y_true, out)


def test_voigt_peak():
    y_true = [
        0.03248735,
        0.04030525,
        0.05136683,
        0.06778597,
        0.09377683,
        0.13884921,
        0.22813635,
        0.43385822,
        0.90715199,
        1.65795663,
        2.08709281,
        1.65795663,
        0.90715199,
        0.43385822,
        0.22813635,
        0.13884921,
        0.09377683,
        0.06778597,
        0.05136683,
        0.04030525,
    ]

    x = np.arange(-1, 1, 0.1)
    a = 1
    cen = 0
    std = 0.1

    out1 = voigt(x, a, cen, std, gamma=0.1)
    out2 = voigt(x, a, cen, std)

    assert_array_almost_equal(y_true, out1)
    assert_array_almost_equal(y_true, out2)


def test_pvoigt_peak():
    y_true = [
        0.01575792,
        0.01940914,
        0.02448538,
        0.03183099,
        0.04301488,
        0.06122087,
        0.09428971,
        0.18131419,
        0.58826472,
        2.00562834,
        3.58626083,
        2.00562834,
        0.58826472,
        0.18131419,
        0.09428971,
        0.06122087,
        0.04301488,
        0.03183099,
        0.02448538,
        0.01940914,
    ]

    x = np.arange(-1, 1, 0.1)
    a = 1
    cen = 0
    std = 0.1
    fraction = 0.5

    out = pvoigt(x, a, cen, std, fraction)

    assert_array_almost_equal(y_true, out)


def test_elastic_model():
    area = 11
    energy = 10
    offset = 0.02
    fanoprime = 0.03
    e_offset = 0
    e_linear = 0.01
    e_quadratic = 0

    true_param = [fanoprime, area, energy]

    x = np.arange(800, 1200, 1)
    out = elastic(x, area, energy, offset, fanoprime, e_offset, e_linear, e_quadratic)

    elastic_model = ElasticModel()

    # fwhm_offset is not a sensitive parameter, used as a fixed value
    elastic_model.set_param_hint(name="e_offset", value=0, vary=False)
    elastic_model.set_param_hint(name="e_linear", value=0.01, vary=False)
    elastic_model.set_param_hint(name="e_quadratic", value=0, vary=False)
    elastic_model.set_param_hint(name="coherent_sct_energy", value=10, vary=False)
    elastic_model.set_param_hint(name="fwhm_offset", value=0.02, vary=False)
    elastic_model.set_param_hint(name="fwhm_fanoprime", value=0.03, vary=False)

    result = elastic_model.fit(out, x=x, coherent_sct_amplitude=10)

    fitted_val = [
        result.values["fwhm_fanoprime"],
        result.values["coherent_sct_amplitude"],
        result.values["coherent_sct_energy"],
    ]

    assert_array_almost_equal(true_param, fitted_val, decimal=2)


def test_compton_model():
    energy = 10
    offset = 0.001
    fano = 0.01
    angle = 90
    fwhm_corr = 1
    amp = 20
    f_step = 0.05
    f_tail = 0.1
    gamma = 2
    hi_f_tail = 0.01
    hi_gamma = 1
    e_offset = 0
    e_linear = 0.01
    e_quadratic = 0
    x = np.arange(800, 1200, 1.0)

    true_param = [energy, amp]

    out = compton(
        x,
        amp,
        energy,
        offset,
        fano,
        e_offset,
        e_linear,
        e_quadratic,
        angle,
        fwhm_corr,
        f_step,
        f_tail,
        gamma,
        hi_f_tail,
        hi_gamma,
    )

    cm = ComptonModel()
    # parameters not sensitive
    cm.set_param_hint(name="compton_hi_gamma", value=hi_gamma, vary=False)
    cm.set_param_hint(name="fwhm_offset", value=offset, vary=False)
    cm.set_param_hint(name="compton_angle", value=angle, vary=False)
    cm.set_param_hint(name="e_offset", value=e_offset, vary=False)
    cm.set_param_hint(name="e_linear", value=e_linear, vary=False)
    cm.set_param_hint(name="e_quadratic", value=e_quadratic, vary=False)
    cm.set_param_hint(name="fwhm_fanoprime", value=fano, vary=False)
    cm.set_param_hint(name="compton_hi_f_tail", value=hi_f_tail, vary=False)
    cm.set_param_hint(name="compton_f_step", value=f_step, vary=False)
    cm.set_param_hint(name="compton_f_tail", value=f_tail, vary=False)
    cm.set_param_hint(name="compton_gamma", value=gamma, vary=False)
    cm.set_param_hint(name="compton_amplitude", value=20, vary=False)
    cm.set_param_hint(name="compton_fwhm_corr", value=fwhm_corr, vary=False)

    p = cm.make_params()
    result = cm.fit(out, x=x, params=p, compton_amplitude=20, coherent_sct_energy=10)

    fit_val = [result.values["coherent_sct_energy"], result.values["compton_amplitude"]]

    assert_array_almost_equal(true_param, fit_val, decimal=2)


def test_dist():
    M = 1.9  # number of coherent modes
    K = 3.15  # number of photons

    bin_edges = np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0])

    pk_n = nbinom_dist(bin_edges, K, M)

    pk_p = poisson_dist(bin_edges, K)

    pk_g = gamma_dist(bin_edges, K, M)

    assert_array_almost_equal(
        pk_n, np.array([0.15609113, 0.17669628, 0.18451672, 0.1837303, 0.17729389, 0.16731627])
    )
    assert_array_almost_equal(pk_g, np.array([0.0, 0.13703903, 0.20090424, 0.22734693, 0.23139384, 0.22222281]))
    assert_array_almost_equal(
        pk_p, np.array([0.04285213, 0.07642648, 0.11521053, 0.15411372, 0.18795214, 0.21260011])
    )
