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

logger = logging.getLogger(__name__)


def negative_binom_distribution(bin_edges, K, M):
    """
    Negative Binomial (Poisson-Gamma) distribution function

    Parameters
    ----------
    bin_edges : array
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
        P(K) = \frac{\\Gamma(K + M)} {\\Gamma(K + 1) ||Gamma(M)}(\frac {M} {M + <K>})^M (\frac {<K>}{M + <K>})^K

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
    co_eff = np.exp(gammaln(bin_edges + M) -
                    gammaln(bin_edges + 1) - gammaln(M))

    poisson_gamma = co_eff * np.power(M / (K + M), M) * np.power(K / (M + K),
                                                                 bin_edges)
    return poisson_gamma


def poisson_distribution(bin_edges, K):
    """
    Poisson Distribution

    Parameters
    ---------
    K : int
        number of photons

    bin_edges : array
        normalized speckle count bin centers

    Returns
    -------
    poisson_dist : array
       Poisson Distribution

    Notes
    -----
    These implementations are based on the references under
    nbinom_distribution() function Notes

    :math ::
        P(K) = \frac{<K>^K}{K!}\exp(-K)
    """
    poisson_dist = np.exp(-K) * np.power(K, bin_edges)/gamma(bin_edges + 1)
    return poisson_dist


def gamma_distribution(bin_edges, K, M):
    """
    Gamma distribution function

    Parameters
    ----------
    bin_edges : array
        normalized speckle count bin centers

    K : int
        number of photons

    M : int
        number of coherent modes

    Returns
    -------
    gamma_dist : array
        Gamma distribution

    Notes
    -----
    These implementations are based on the references under
    negative_binom_distribution() function Notes

    : math ::
        P(K) =(\frac{M}{<K>})^M \frac{K^(M-1)}{\Gamma(M)}\exp(-M\frac{K}{<K>})

    """
    co_eff = np.exp(M * np.log(M) + (M - 1) * np.log(bin_edges) -
                    gammaln(M) - M * np.log(K))
    gamma_dist = co_eff * np.exp(- M * bin_edges / K)
    return gamma_dist
