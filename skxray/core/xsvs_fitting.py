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


def negative_binom_distribution(K, M, bin_centers):
    """
    Negative Binomial (Poisson-Gamma) distribution function
    Parameters
    ----------
    K : int
        number of photons

    M : int
        number of coherent modes

    bin_centers : array
        normalized speckle count bin centers

    Returns
    -------
    poisson_dist : array
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
    co_eff = np.exp(gammaln(bin_centers + M) -
                    gammaln(bin_centers + 1) - gammaln(M))

    poisson_gamma = co_eff * np.power(M / (K + M), M)
    co_eff2 = np.power(K / (M + K), bin_centers)
    poisson_gamma *= co_eff2
    return poisson_gamma


def poisson_distribution(K, bin_centers):
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

    Note
    ----
    These implementations are based on the references under
    nbinom_distribution() function Note

    """
    poisson_dist = np.exp(-K) * np.power(K, bin_centers)/gamma(bin_centers + 1)
    return poisson_dist


def gamma_distribution(M, K, bin_centers):
    """
    Gamma distribution function

    Parameters
    ----------
    M : int
        number of coherent modes

    K : int
        number of photons

    bin_centers : array
        normalized speckle count bin centers

    Returns
    -------
    gamma_dist : array
        Gamma distribution

    Note
    ----
    These implementations are based on the references under
    negative_binom_distribution() function Note
    """

    co_eff = np.exp(M * np.log(M) + (M - 1) * np.log(bin_centers) -
                   gammaln(M) - M * np.log(K))
    gamma_dist = co_eff * np.exp(- M * bin_centers / K)
    return gamma_dist


def model_residuals(params, bin_centers, y, yerr):
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
        probability of detecting speckles

    bin_centers : array
        normalized speckle count bin centers

    yerr : array, optional
        standard error of y

    Returns
    -------
    model_residual : array
        Residuals function for least squares fitting

    Note
    ----
    These implementations are based on the references under
    negative_binom_distribution() function Note
    """
    # create set of parameters
    M = params['M'].value
    K = params['K'].value

    return (y - negative_binom_distribution(K, M, bin_centers))/yerr


def eval_binomal_dist(params, bin_centers, data, err):
    """
    Function will minimize difference between probability of the detecting
    speckles and negative binomial distribution for the given set of
    input parameters.

    Parameters
    ----------
    M : int
        number of coherent modes

    K : int
        average number of photons

    bin_centers : array
        normalized speckle count bin centers

    data : array
        probability of detecting speckles

    err : array
        standard error of data

    Returns
    -------
    final_result : array

    Note
    ----
    These implementations are based on the references under
    negative_binom_distribution() function Note

    """
    result = minimize(model_residuals, params, args=(bin_centers, data, err))
    return data + result.residual


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

    contrast_factor : float

    cf_baseline : float, optional

    Return
    ------
    diff_contrast_factor : array
        speckle contrast factor for samples undergoing a diffusive motion

    Note
    ----
    integration times more information - geometric_series function in
    skxray.core module

    These implementations are based on the references under
    negative_binom_distribution() function Note

    """
    co_eff = (np.exp(-2*relaxation_rate*times) - 1 +
              2*relaxation_rate*times)/(2*(relaxation_rate*times)**2)

    return contrast_factor*co_eff + cf_baseline


def cf_residuals(params, times, data, err):
    """
    The dynamic information of the sample motions extracted by fitting
    experimental speckle contrast factor.

    Parameters
    ----------
    params : dict or Parameters
        relaxation_rate - float, relaxation time associated with the
        samples dynamics,
        contrast_factor - float, optical contrast (speckle contrast),
        cf_baseline - float, baseline of contrast factor (usually ~zero)
        Have to give either as a dictionary or Parameters
        # create a dictionary
        {'relaxation_rate': 6.7, 'contrast_factor': 0.23, 'cf_baseline':1}

        # create a set of Parameters
        from lmfit import Parameters
        params = Parameters()
        params.add('contrast_factor',  value=0.22,  min=0.19, max=0.0.23)
        params.add('relaxation_rate', value=6.7, min=6.68, max=6.74)
        params.add('cf_baseline', value=0)

    times : array
        integration times

    data : array
        contrast factor for sample undergoing diffusive motion

    err : array
        standard error of the contrast factor

    Returns
    --------
    model_residual : array
        Residuals function for least squares fitting

    Note
    ----
    integration times more information - geometric_series function in
    skxray.core module

    These implementations are based on the references under
    negative_binom_distribution() function Note

    """
    # create set of parameters
    relax_rate = params['relaxation_rate'].value
    cf = params['contrast_factor'].value
    cf_baseline = params['cf_baseline'].value

    return (data - diffusive_motion_contrast_factor(times, relax_rate,
                                                    cf, cf_baseline))


def minimize_dm_cf(params, times, data, err):
    """

    Parameters
    ----------
    params :dict or Parameters
        relaxation_rate - float, relaxation time associated with the
        samples dynamics,
        contrast_factor - float, optical contrast (speckle contrast),
        cf_baseline - float, baseline of contrast factor (usually ~zero)
        Have to give either as a dictionary or Parameters
        # create a dictionary
        {'relaxation_rate': 6.7, 'contrast_factor': 0.23, 'cf_baseline':1}

        # create a set of Parameters
        from lmfit import Parameters
        params = Parameters()
        params.add('contrast_factor',  value=0.22,  min=0.19, max=0.0.23)
        params.add('relaxation_rate', value=6.7, min=6.68, max=6.74)
        params.add('cf_baseline', value=0)

    times : array
        integration times

    data : array
        contrast factor for sample undergoing diffusive motion

    err : array
        standard error of the contrast factor

    Returns
    -------
     final_result : array
        minimized fitting to results using least square model

    Note
    ----
    integration times more information - geometric_series function in
    skxray.core module

    These implementations are based on the references under
    negative_binom_distribution() function Note

    """
    result = minimize(cf_residuals, params, args=(times, data, err))
    return data + result.residual


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

    Note
    ----
    These implementations are based on the references under
    negative_binom_distribution() function Note

    """
    return relaxation_rates/(q_values**2)


def diff_co_residuals(params, diff_co, q_values):
    """
    Parameters
    ----------
    params : dict or Parameters
        relax_rate - float, relaxation time associated with the
        samples dynamics,

    diff_co : array
        diffusive coefficients

    q_values:

    Return
    ------

    """
    # create set of parameters
    relax_rate = params['relaxation_rate'].value

    return (diff_co - diffusive_coefficient(relax_rate, q_values))


def minimize_diff_co(params, q_values,  diff_co, err):

    result = minimize(diffusive_coefficient, params, args=(1/q_values**2,
                                                           diff_co, err))
    return diff_co + result.residual
