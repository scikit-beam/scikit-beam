'''
Copyright (c) 2014, Brookhaven National Laboratory
All rights reserved.

# @author: Li Li (lili@bnl.gov)
# created on 07/10/2014

Original code:
@author: Mirna Lerotic, 2nd Look Consulting
         http://www.2ndlookconsulting.com/
Copyright (c) 2013, Stefan Vogt, Argonne National Laboratory
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the Brookhaven National Laboratory nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


import numpy as np
import scipy.special


def model_gauss_peak(area, sigma, dx):
    """
    model a gaussian fluorescence peak
    
    Parameters
    ----------
    area : float
        area of gaussian function
    sigma : float
        standard deviation
    x : array
        data in x coordinate, relative to center
        
    Returns
    -------
    array
        gaussian peak

    References
    ----------
    .. [1] Rene Van Grieken, "Handbook of X-Ray Spectrometry, Second Edition,
           (Practical Spectroscopy)", CRC Press, 2 edition, pp. 182, 2007.
    
    """
    
    return area / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((dx / sigma)**2))


def model_gauss_step(area, sigma, dx, peak_e):
    """
    use scipy erfc function
    erfc = 1-erf
    
    Parameters
    ----------
    area : float
        area of gauss step function
    sigma : float
        standard deviation
    dx : array
        data in x coordinate, x > 0
    peak_e : float
        need to double check this value
    
    Returns
    -------
    counts : array
        gaussian step peak

    References
    ----------
    .. [1] Rene Van Grieken, "Handbook of X-Ray Spectrometry, Second Edition,
           (Practical Spectroscopy)", CRC Press, 2 edition, pp. 182, 2007.
    """
    
    counts = area / 2. / peak_e * scipy.special.erfc(dx / (np.sqrt(2) * sigma))

    return counts


def model_gauss_tail(area, sigma, dx, gamma):
    """
    models a gaussian tail function
    refer to van espen, spectrum evaluation,
    in van grieken, handbook of x-ray spectrometry, 2nd ed, page 182
    
    Parameters
    ----------
    area : float
        area of gauss tail function
    sigma : float
        control peak width
    dx : array
        data in x coordinate
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

    dx_neg = np.array(dx)
    dx_neg[dx_neg > 0] = 0
    
    temp_a = np.exp(dx_neg / (gamma * sigma))
    counts = area / (2 * gamma * sigma * np.exp(-0.5 / (gamma**2))) * \
        temp_a * scipy.special.erfc(dx / (np.sqrt(2) * sigma) + (1 / (gamma*np.sqrt(2))))

    return counts


def elastic_peak(coherent_sct_energy,
                 fwhm_offset, fwhm_fanoprime,
                 area, ev, epsilon=2.96):
    """
    model elastic peak as a gaussian function
    
    Parameters
    ----------
    coherent_sct_energy : float
        incident energy                         
    fwhm_offset : float
        global parameter for peak width    
    fwhm_fanoprime : float
        global parameter for peak width
    area : float:
        area of gaussian peak
    ev : array
        energy value
    epsilon : float
        energy to create a hole-electron pair
        for Ge 2.96, for Si 3.61 at 300K
        needs to double check this value
    
    Returns
    -------
    value : array
        elastic peak
    sigma : float
        peak width
                     
    """
    
    temp_val = 2 * np.sqrt(2 * np.log(2))
    sigma = np.sqrt((fwhm_offset / temp_val)**2 +
                    coherent_sct_energy * epsilon * fwhm_fanoprime)
    
    delta_energy = ev - coherent_sct_energy

    value = model_gauss_peak(area, sigma, delta_energy)
    
    return value, sigma


def compton_peak(coherent_sct_energy, fwhm_offset, fwhm_fanoprime, 
                 compton_angle, compton_fwhm_corr, compton_amplitude,
                 compton_f_step, compton_f_tail, compton_gamma,
                 compton_hi_f_tail, compton_hi_gamma,
                 area, ev, epsilon=2.96, matrix=False):
    """
    model compton peak
    
    Parameters
    ----------
    coherent_sct_energy : float
        incident energy                         
    fwhm_offset : float
        global fitting parameter for peak width
    fwhm_fanoprime : float
        global fitting parameter for peak width
    compton_angle : float
        compton angle
    compton_fwhm_corr : float 
        correction factor on peak width
    compton_amplitude : float
        amplitude of compton peak
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
    area : float
        area for gaussian peak, gaussian step and gaussian tail functions
    ev : array
        energy value
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
    sigma : float
        standard deviation
    factor : float
        weight factor of gaussian peak

    References
    ----------
    .. [1] M. Van Gysel etc, "Description of Compton peaks in energy-dispersive
           x-ray ﬂuorescence spectra", X-Ray Spectrometry, vol. 32, pp. 139–147, 2003.
    """
    
    compton_e = coherent_sct_energy / (1 + (coherent_sct_energy / 511) *
                                       (1 - np.cos(compton_angle * np.pi / 180)))
    
    temp_val = 2 * np.sqrt(2 * np.log(2))
    sigma = np.sqrt((fwhm_offset / temp_val)**2 + compton_e * epsilon * fwhm_fanoprime)
    
    #local_sigma = sigma*p[14]

    delta_energy = ev.copy() - compton_e

    counts = np.zeros(len(ev))

    factor = 1 / (1 + compton_f_step + compton_f_tail + compton_hi_f_tail)
    
    if matrix is False:
        factor = factor * (10.**compton_amplitude)
        
    value = factor * model_gauss_peak(area, sigma*compton_fwhm_corr, delta_energy)
    counts += value

    # compton peak, step
    if compton_f_step > 0.:
        value = factor * compton_f_step
        value *= model_gauss_step(area, sigma, delta_energy, compton_e)
        counts += value
    
    # compton peak, tail on the low side
    value = factor * compton_f_tail
    value *= model_gauss_tail(area, sigma, delta_energy, compton_gamma)
    counts += value

    # compton peak, tail on the high side
    value = factor * compton_hi_f_tail
    value *= model_gauss_tail(area, sigma, -1. * delta_energy, compton_hi_gamma)
    counts += value

    return counts, sigma, factor
