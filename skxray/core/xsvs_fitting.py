# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Original code:                                                       #
# @author: Andrei Fluerasu, Brookhaven National Laboratory and         #
# Pawel Kwasniewski, European Synchrotron Radiation Facility           #
#                                                                      #
# Developed at the NSLS-II, Brookhaven National Laboratory             #
# Developed by Sameera K. Abeykoon, June 2015                          #
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
    This module will provide fitting tools for
    X-ray Speckle Visibility Spectroscopy (XSVS)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six

from six.moves import zip
from six import string_types

import logging

import numpy as np
from time import time

from scipy.stats import nbinom
from scipy.special import gamma, gammaln

from lmfit import minimize, Parameters

logger = logging.getLogger(__name__)


def negative_binom_distribution(bin_centers, K, M):
    """
    Negative Binomial (Poisson-Gamma) distribution function

    Parameters
    ----------
    bin_centers : array
        normalized speckle count bin centers

    K : int
        number of photons

    M : int
        number of coherent modes

    Returns
    -------
    poisson_gamma : array
        Negative Binomial (Poisson-Gamma) distribution function

    Notes
    -----
    The negative-binomial distribution function

    :math ::
        P(K) = (\frac{\\Gamma(K + M)} {\\Gamma(K + 1) ||Gamma(M)}(\frac {M} {M + <K>})^M (\frac {<K>}{M + <K>})^K

    These implementation is based on following references
    References: text [1]_, text [2]_

    .. [1] L. Li, P. Kwasniewski, D. Oris, L Wiegart, L. Cristofolini,
       C. Carona and A. Fluerasu , "Photon statistics and speckle visibility
       spectroscopy with partially coherent x-rays" J. Synchrotron Rad.,
       vol 21, p 1288-1295, 2014.

    .. [2] R. Bandyopadhyay, A. S. Gittings, S. S. Suh, P.K. Dixon and
       D.J. Durian "Speckle-visibilty Spectroscopy: A tool to study
       time-varying dynamics" Rev. Sci. Instrum. vol 76, p  093110, 2005.

    """
    co_eff = np.exp(gammaln(bin_centers + M) -
                    gammaln(bin_centers + 1) - gammaln(M))

    poisson_gamma = co_eff * np.power(M / (K + M), M)
    co_eff2 = np.power(K / (M + K), bin_centers)
    poisson_gamma *= co_eff2
    return poisson_gamma


def poisson_distribution(bin_centers, K):
    """
    Poisson Distribution

    Parameters
    ---------
    K : int
        number of photons

    bin_centers : array
        normalized speckle count bin centers

    Returns
    -------
    poisson_dist : array
       Poisson Distribution

    Notes
    -----
    These implementations are based on the references under
    nbinom_distribution() function Notes

    """
    poisson_dist = np.exp(-K) * np.power(K, bin_centers)/gamma(bin_centers + 1)
    return poisson_dist


def gamma_distribution(bin_centers, M, K):
    """
    Gamma distribution function

    Parameters
    ----------
    bin_centers : array
        normalized speckle count bin centers

    M : int
        number of coherent modes

    K : int
        number of photons

    Returns
    -------
    gamma_dist : array
        Gamma distribution

    Notes
    -----
    These implementations are based on the references under
    negative_binom_distribution() function Notes
    """

    co_eff = np.exp(M * np.log(M) + (M - 1) * np.log(bin_centers) -
                   gammaln(M) - M * np.log(K))
    gamma_dist = co_eff * np.exp(- M * bin_centers / K)
    return gamma_dist


def diffusive_coefficient(relaxation_rates, q_values):
    """
    For Brownian samples, the diffusive coefficient can be obtained

    Parameters
    ---------
    relaxation_rates : array
        relaxation rates of the sample Brownian motion

    q_values : array
        scattering vectors for each relaxation rates
        (same shape as relaxation_rates)

    Returns
    -------
    diff_co : float
        diffusive coefficient for Brownian samples

    Notes
    -----
    These implementations are based on the references under
    negative_binom_distribution() function Notes

    """
    return relaxation_rates/(q_values**2)


def diffusive_motion_contrast_factor(times, relaxation_rate,
                                     contrast_factor, cf_baseline=0):
    """
    This will provide the speckle contrast factor of samples undergoing
    a diffusive motion.

    Parameters
    ----------
    times : array
        integration times

    relaxation_rate : float
        relaxation rate

    contrast_factor : float
        contrast factor

    cf_baseline : float, optional
        the baseline for the contrast factor

    Return
    ------
    diff_contrast_factor : array
        speckle contrast factor for samples undergoing a diffusive motion

    Notes
    -----
    integration times more information - geometric_series function in
    skxray.core.utils module

    These implementations are based on the references under
    negative_binom_distribution() function Notes

    """
    co_eff = (np.exp(-2*relaxation_rate*times) - 1 +
              2*relaxation_rate*times)/(2*(relaxation_rate*times)**2)

    return contrast_factor*co_eff + cf_baseline
