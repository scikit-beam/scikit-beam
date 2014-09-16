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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


import numpy as np
import scipy.special


def gauss_peak(x, area, center, sigma):
    """
    Use gaussian function to model fluorescence peak from each element
    
    Parameters
    ----------
    x : array
        data in x coordinate
    area : float
        area of gaussian function
    center : float
        center position
    sigma : float
        standard deviation
        
    Returns
    -------
    couunts : ndarray
        gaussian peak

    References
    ----------
    .. [1] Rene Van Grieken, "Handbook of X-Ray Spectrometry, Second Edition,
           (Practical Spectroscopy)", CRC Press, 2 edition, pp. 182, 2007.
    
    """
    
    return area / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (((x-center) / sigma)**2))


def gauss_step(x, area, center, sigma, peak_e):
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
    .. [1] Rene Van Grieken, "Handbook of X-Ray Spectrometry, Second Edition,
           (Practical Spectroscopy)", CRC Press, 2 edition, pp. 182, 2007.
    """
    
    return area / 2. / peak_e * scipy.special.erfc((x - center) / (np.sqrt(2) * sigma))


def gauss_tail(x, area, center, sigma, gamma):
    """
    Use a gaussian tail function to simulate compton peak
    
    Parameters
    ----------
    x : array
        data in x coordinate
    area : float
        area of gauss tail function
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
    .. [1] Rene Van Grieken, "Handbook of X-Ray Spectrometry, Second Edition,
           (Practical Spectroscopy)", CRC Press, 2 edition, pp. 182, 2007.
    """

    dx_neg = np.array(x) - center
    dx_neg[dx_neg > 0] = 0
    
    temp_a = np.exp(dx_neg / (gamma * sigma))
    counts = area / (2 * gamma * sigma * np.exp(-0.5 / (gamma**2))) * \
        temp_a * scipy.special.erfc((x - center) / (np.sqrt(2) * sigma) + (1 / (gamma * np.sqrt(2))))

    return counts


def elastic_peak(x, coherent_sct_energy,
                 fwhm_offset, fwhm_fanoprime,
                 coherent_sct_amplitude, epsilon=2.96):
    """
    Use gaussian function to model elastic peak
    
    Parameters
    ----------
    x : array
        energy value
    coherent_sct_energy : float
        incident energy                         
    fwhm_offset : float
        global fitting parameter for peak width
    fwhm_fanoprime : float
        global fitting parameter for peak width
    coherent_sct_amplitude : float
        area of gaussian peak
    epsilon : float
        energy to create a hole-electron pair
        for Ge 2.96, for Si 3.61 at 300K
        needs to double check this value
    
    Returns
    -------
    value : array
        elastic peak
                     
    """
    
    temp_val = 2 * np.sqrt(2 * np.log(2))
    sigma = np.sqrt((fwhm_offset / temp_val)**2 +
                    coherent_sct_energy * epsilon * fwhm_fanoprime)

    value = gauss_peak(x, coherent_sct_amplitude, coherent_sct_energy, sigma)
    
    return value


def compton_peak(x, coherent_sct_energy, fwhm_offset, fwhm_fanoprime,
                 compton_angle, compton_fwhm_corr, compton_amplitude,
                 compton_f_step, compton_f_tail, compton_gamma,
                 compton_hi_f_tail, compton_hi_gamma,
                 epsilon=2.96, matrix=False):
    """
    Model compton peak, which is generated as an inelastic peak and always
    stays to the left of elastic peak on the spectrum.
    
    Parameters
    ----------
    x : array
        energy value
    coherent_sct_energy : float
        incident energy                         
    fwhm_offset : float
        global fitting parameter for peak width
    fwhm_fanoprime : float
        global fitting parameter for peak width
    compton_angle : float
        compton angle in degree
    compton_fwhm_corr : float 
        correction factor on peak width
    compton_amplitude : float
        area for gaussian peak, gaussian step and gaussian tail functions
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
    matrix : bool
        to be updated
    
    Returns
    -------
    counts : array
        compton peak

     References
    -----------
    .. [1] M. Van Gysel etc, "Description of Compton peaks in energy-dispersive x-ray fluorescence spectra",
           X-Ray Spectrometry, vol. 32, pp. 139-147, 2003.
    """
    compton_e = coherent_sct_energy / (1 + (coherent_sct_energy / 511) *
                                       (1 - np.cos(compton_angle * np.pi / 180)))
    
    temp_val = 2 * np.sqrt(2 * np.log(2))
    sigma = np.sqrt((fwhm_offset / temp_val)**2 + compton_e * epsilon * fwhm_fanoprime)

    counts = np.zeros_like(x)

    factor = 1 / (1 + compton_f_step + compton_f_tail + compton_hi_f_tail)
    
    if matrix is False:
        factor = factor * (10.**compton_amplitude)
        
    value = factor * gauss_peak(x, compton_amplitude, compton_e, sigma*compton_fwhm_corr)
    counts += value

    # compton peak, step
    if compton_f_step > 0.:
        value = factor * compton_f_step
        value *= gauss_step(x, compton_amplitude, compton_e, sigma, compton_e)
        counts += value
    
    # compton peak, tail on the low side
    value = factor * compton_f_tail
    value *= gauss_tail(x, compton_amplitude, compton_e, sigma, compton_gamma)
    counts += value

    # compton peak, tail on the high side
    value = factor * compton_hi_f_tail
    value *= gauss_tail(-1 * x, compton_amplitude, -1 * compton_e, sigma, compton_hi_gamma)
    counts += value

    return counts
