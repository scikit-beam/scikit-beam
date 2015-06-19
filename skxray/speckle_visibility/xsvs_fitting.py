# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Original code:                                                       #
# @author: Pawel Kwasniewski, European Synchrotron Radiation Facility  #
# and Andrei Fluerasu, Brookhaven                                      #
#                                                                      #
# Developed at the NSLS-II, Brookhaven National Laboratory             #
# Developed by Sameera K. Abeykoon, May 2015                           #
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
logger = logging.getLogger(__name__)

import numpy as np
from time import time

from ConfigParser import RawConfigParser
from os.path import isfile
import os
from sys import argv, stdout
import sys

from scipy.stats import nbinom
from scipy.optimize import leastsq
from scipy.special import gamma, gammaln


def nbinom_distribution(K, M, x):
    """
    Negative Binomial (Poisson-Gamma) distribution function
    Parameters
    ----------
    K : int
        number of photons

    M : int
        number of coherent modes

    x : array

    Returns
    -------
    Pk : array
        Negative Binomial (Poisson-Gamma) distribution function

    Note
    ----
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
    coeff = np.exp(gammaln(x + M) - gammaln(x + 1) - gammaln(M))

    Poission_Gamma = coeff * np.power(M / (K + M), M)
    coeff2 = np.power(K / (M + K), x)
    Poission_Gamma *= coeff2
    return Poission_Gamma


def poisson_distribution(K, x):
    """
    Poisson Distribution

    Parameters
    ---------
    K : int
        number of photons

    x : array

    Returns
    -------
    Poission_D : array
       Poisson Distribution

    Note
    ----
    These implementation based on the references under
    nbinom_distribution() function Note

    """
    Poission_D = np.exp(-K) * np.power(K, x)/gamma(x + 1)
    return Poission_D


def gamma_distribution(M, K, x ):
    """
    Gamma distribution function

    Parameters
    ----------
    M : int
        number of coherent modes

    K : int
        number of photons

    x : array

    Returns
    -------
    G : array
        Gamma distribution

    Note
    ----
    These implementation based on the references under
    nbinom_distribution() function Note
    """

    coeff = np.exp(M * np.log(M) + (M - 1) * np.log(x) -
                   gammaln(M) - M * np.log(K))
    Gd = coeff * np.exp(- M * x / K)
    return Gd


def residuals(M, K, y, x, yerr):
    """
    Residuals function for least squares fitting,
    K may be a fixed parameter

    Parameters
    ----------
    M : int
        number of coherent modes

    K : int
        number of photons

    y : array

    x : array

    yerr : array

    Returns
    -------
    residual : array
        Residuals function for least squares fitting

    Note
    ----
    These implementation based on the references under
    nbinom_distribution() function Note
    """
    pr = M / (K + M)

    residual = (y - np.log10(nbinom_distribution(x, K, M)))/yerr
    return residual


def eval_binomal_dist(M, K, x):
    """
    Function evaluating the binomial distribution for the given set of
    input parameters. Redundant - should be removed.
    Parameters
    ----------
    M : int
        number of coherent modes

    K : int
        average number of photons

    x : array


    Returns
    -------
    eval_result : array

    Note
    ----
    These implementation based on the references under
    nbinom_distribution() function Note
    """

    eval_result = nbinom_distribution(x, K, M)
    return eval_result
