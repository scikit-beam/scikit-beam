# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# @author: Li Li (lili@bnl.gov)                                        #
# created on 07/10/2014                                                #
#                                                                      #
# Original code:                                                       #
# @author: Mirna Lerotic, 2nd Look Consulting                          #
#         http://www.2ndlookconsulting.com/                            #
# Copyright (c) 2013, Stefan Vogt, Argonne National Laboratory         #
# All rights reserved.                                                 #
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

import logging

import numpy as np
import scipy.special
from scipy import stats
from scipy.special import gamma, gammaln

logger = logging.getLogger(__name__)


log2 = np.log(2)
s2pi = np.sqrt(2 * np.pi)
spi = np.sqrt(np.pi)
s2 = np.sqrt(2.0)


def gaussian(x, area, center, sigma):
    """1 dimensional gaussian

    Parameters
    ----------
    x : array
        independent variable
    area : float
        Area of the normally distributed peak
    center : float
        center position
    sigma : float
        standard deviation
    """
    return (area / (s2pi * sigma)) * np.exp(-1 * (1.0 * x - center) ** 2 / (2 * sigma**2))


def lorentzian(x, area, center, sigma):
    """1 dimensional lorentzian

    Parameters
    ----------
    x : array
        independent variable
    area : float
        area of lorentzian peak,
        If area is set as 1, the integral is unity.
    center : float
        center position
    sigma : float
        standard deviation
    """
    return (area / (1 + ((1.0 * x - center) / sigma) ** 2)) / (np.pi * sigma)


def lorentzian2(x, area, center, sigma):
    """1-d lorentzian squared profile

    Parameters
    ----------
    x : array
        independent variable
    area : float
        area of lorentzian squared peak,
        If area is set as 1, the integral is unity.
    center : float
        center position
    sigma : float
        standard deviation
    """

    return (area / (1 + ((x - center) / sigma) ** 2) ** 2) / (np.pi * sigma)


def voigt(x, area, center, sigma, gamma=None):
    """Convolution of gaussian and lorentzian curve.

    see http://en.wikipedia.org/wiki/Voigt_profile

    Parameters
    ----------
    x : array
        independent variable
    area : float
        area of voigt peak
    center : float
        center position
    sigma : float
        standard deviation
    gamma : float, optional
        half width at half maximum of lorentzian.
        If optional, `gamma` gets set to `sigma`
    """
    if gamma is None:
        gamma = sigma
    z = (x - center + 1j * gamma) / (sigma * s2)
    return area * scipy.special.wofz(z).real / (sigma * s2pi)


def pvoigt(x, area, center, sigma, fraction):
    """Linear combination  of gaussian and lorentzian

    Parameters
    ----------
    x : array
        independent variable
    area : float
        area of pvoigt peak
    center : float
        center position
    sigma : float
        standard deviation
    fraction : float
        weight for lorentzian peak in the linear combination, and (1-fraction)
        is the weight for gaussian peak.
    """
    return (1 - fraction) * gaussian(x, area, center, sigma) + fraction * lorentzian(x, area, center, sigma)


def gausssian_step(x, area, center, sigma, peak_e):
    """
    Gauss step function is an important component in modeling compton peak.
    Use scipy erfc function. Please note erfc = 1-erf.

    Parameters
    ----------
    x : array
        data in x coordinate
    area : float
        area of gauss step function
    center : float
        center position
    sigma : float
        standard deviation
    peak_e : float
        emission energy

    Returns
    -------
    counts : array
        gaussian step peak

    References
    ----------
    .. [1]
        Rene Van Grieken, "Handbook of X-Ray Spectrometry, Second Edition,
        (Practical Spectroscopy)", CRC Press, 2 edition, pp. 182, 2007.
    """

    return area * scipy.special.erfc((x - center) / (np.sqrt(2) * sigma)) / (2.0 * peak_e)


def gaussian_tail(x, area, center, sigma, gamma):
    """
    Use a gaussian tail function to simulate compton peak

    Parameters
    ----------
    x : array
        data in x coordinate
    area : float
        area of gauss tail function
        If area is set as 1, the integral is unity.
    center : float
        center position
    sigma : float
        control peak width
    gamma : float
        normalization factor

    Returns
    -------
    counts : array
        gaussian tail peak

    References
    ----------
    .. [1]
        Rene Van Grieken, "Handbook of X-Ray Spectrometry, Second Edition,
        (Practical Spectroscopy)", CRC Press, 2 edition, pp. 182, 2007.
    """

    dx_neg = np.array(x) - center
    dx_neg[dx_neg > 0] = 0

    temp_a = np.exp(dx_neg / (gamma * sigma))

    v1 = scipy.special.erfc((x - center) / (np.sqrt(2) * sigma) + (1 / (gamma * np.sqrt(2))))
    v2 = 2 * gamma * sigma * np.exp(-0.5 / (gamma**2))
    counts = area * temp_a * (v1 / v2)

    return counts


def elastic(
    x,
    coherent_sct_amplitude,
    coherent_sct_energy,
    fwhm_offset,
    fwhm_fanoprime,
    e_offset,
    e_linear,
    e_quadratic,
    epsilon=2.96,
):
    """Model of elastic peak in X-Ray fluorescence

    Parameters
    ----------
    x : array
        energy value
    coherent_sct_amplitude : float
        area of elastic peak
    coherent_sct_energy : float
        incident energy
    fwhm_offset : float
        global fitting parameter for peak width
    fwhm_fanoprime : float
        global fitting parameter for peak width
    e_offset : float
        offset of energy calibration
    e_linear : float
        linear coefficient in energy calibration
    e_quadratic : float
        quadratic coefficient in energy calibration
    epsilon : float
        energy to create a hole-electron pair
        for Ge 2.96, for Si 3.61 at 300K
        needs to double check this value

    Returns
    -------
    value : array
        elastic peak
    """

    x = e_offset + x * e_linear + x**2 * e_quadratic

    temp_val = 2 * np.sqrt(2 * np.log(2))
    sigma = np.sqrt((fwhm_offset / temp_val) ** 2 + coherent_sct_energy * epsilon * fwhm_fanoprime)

    return gaussian(x, coherent_sct_amplitude, coherent_sct_energy, sigma)


def compton(
    x,
    compton_amplitude,
    coherent_sct_energy,
    fwhm_offset,
    fwhm_fanoprime,
    e_offset,
    e_linear,
    e_quadratic,
    compton_angle,
    compton_fwhm_corr,
    compton_f_step,
    compton_f_tail,
    compton_gamma,
    compton_hi_f_tail,
    compton_hi_gamma,
    epsilon=2.96,
):
    """
    Model compton peak, which is generated as an inelastic peak and always
    stays to the left of elastic peak on the spectrum.

    Parameters
    ----------
    x : array
        energy value
    compton_amplitude : float
        area for gaussian peak, gaussian step and gaussian tail functions
    coherent_sct_energy : float
        incident energy
    fwhm_offset : float
        global fitting parameter for peak width
    fwhm_fanoprime : float
        global fitting parameter for peak width
    e_offset : float
        offset of energy calibration
    e_linear : float
        linear coefficient in energy calibration
    e_quadratic : float
        quadratic coefficient in energy calibration
    compton_angle : float
        compton angle in degree
    compton_fwhm_corr : float
        correction factor on peak width
    compton_f_step : float
        weight factor of the gaussian step function
    compton_f_tail : float
        weight factor of gaussian tail on lower side
    compton_gamma : float
        normalization factor of gaussian tail on lower side
    compton_hi_f_tail : float
        weight factor of gaussian tail on higher side
    compton_hi_gamma : float
        normalization factor of gaussian tail on higher side
    epsilon : float
        energy to create a hole-electron pair
        for Ge 2.96, for Si 3.61 at 300K
        needs to double check this value

    Returns
    -------
    counts : array
        compton peak

    References
    ----------
    .. [1]
        M. Van Gysel etc, "Description of Compton peaks in energy-dispersive
        x-ray fluorescence spectra", X-Ray Spectrometry, vol. 32, pp. 139-147,
        2003.
    """

    x = e_offset + x * e_linear + x**2 * e_quadratic

    # the rest-mass energy of an electron (511 keV)
    mc2 = 511
    comp_denom = 1 + coherent_sct_energy / mc2 * (1 - np.cos(np.deg2rad(compton_angle)))
    compton_e = coherent_sct_energy / comp_denom

    temp_val = 2 * np.sqrt(2 * np.log(2))
    sigma = np.sqrt((fwhm_offset / temp_val) ** 2 + compton_e * epsilon * fwhm_fanoprime)

    counts = np.zeros_like(x)

    factor = 1 / (1 + compton_f_step + compton_f_tail + compton_hi_f_tail)

    value = factor * gaussian(x, compton_amplitude, compton_e, sigma * compton_fwhm_corr)
    counts += value

    # compton peak, step
    if compton_f_step > 0.0:
        value = factor * compton_f_step
        value *= gausssian_step(x, compton_amplitude, compton_e, sigma, compton_e)
        counts += value

    # compton peak, tail on the low side
    value = factor * compton_f_tail
    value *= gaussian_tail(x, compton_amplitude, compton_e, sigma, compton_gamma)
    counts += value

    # compton peak, tail on the high side
    value = factor * compton_hi_f_tail
    value *= gaussian_tail(-1 * x, compton_amplitude, -1 * compton_e, sigma, compton_hi_gamma)
    counts += value

    return counts


def gamma_dist(bin_values, K, M):
    """Gamma distribution function

    Parameters
    ----------
    bin_values : array
        bin values for detecting photons
        eg : max photon counts is 8
        bin_values = np.arange(8+2)
    K : int
        mean count of photons
    M : int
        number of coherent modes

    Returns
    -------
    gamma_dist : array
        Gamma distribution

    Notes
    -----
    These implementations are based on the references under the ``Notes``
    section of the ``nbinom_dist()`` docstring

    .. math::
        P(K) = \\frac{\\Gamma(K + M)} {\\Gamma(K + 1)\\Gamma(M)}
        (\\frac {M} {M + <K>})^M (\\frac {<K>}{M + <K>})^K
    """

    gamma_dist = (stats.gamma(M, 0.0, K / M)).pdf(bin_values)
    return gamma_dist


def nbinom_dist(bin_values, K, M):
    """
    Negative Binomial (Poisson-Gamma) distribution function

    Parameters
    ----------
    bin_values : array
        bin values for detecting photons
        eg : max photon counts is 8
        bin_values = np.arange(8+2)
    K : int
        mean count of photons
    M : int
        number of coherent modes

    Returns
    -------
    nbinom : array
        Negative Binomial (Poisson-Gamma) distribution function

    Notes
    -----
    The negative-binomial distribution function

    .. math::
       P(K) =(\\frac{M}{<K>})^M \\frac{K^{M-1}}
       {\\Gamma(M)}\\exp(-M\\frac{K}{<K>})

    Implementation reference [1]_

    References
    ----------
    .. [1]
        L. Li, P. Kwasniewski, D. Oris, L Wiegart, L. Cristofolini,
        C. Carona and A. Fluerasu , "Photon statistics and speckle visibility
        spectroscopy with partially coherent x-rays" J. Synchrotron Rad.,
        vol 21, p 1288-1295, 2014.

    """
    co_eff = np.exp(gammaln(bin_values + M) - gammaln(bin_values + 1) - gammaln(M))

    nbinom = co_eff * np.power(M / (K + M), M) * np.power(K / (M + K), bin_values)
    return nbinom


def poisson_dist(bin_values, K):
    """
    Poisson Distribution

    Parameters
    ----------
    K : int
        mean count of photons
    bin_values : array
        bin values for detecting photons
        eg : max photon counts is 8
        bin_values = np.arange(8+2)

    Returns
    -------
    poisson_dist : array
       Poisson Distribution

    Notes
    -----
    These implementations are based on the references under
    the ``Notes`` section of the ``nbinom_dist()`` docstring

    .. math::
        P(K) = \\frac{<K>^K}{K!}\\exp(-<K>)
    """

    poisson_dist = np.exp(-K) * np.power(K, bin_values) / gamma(bin_values + 1)
    return poisson_dist
