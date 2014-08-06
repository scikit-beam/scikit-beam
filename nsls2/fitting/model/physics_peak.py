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

"""
basic fitting functions used for Fluorescence fitting
"""
import numpy as np
import scipy.special
import scipy.signal


def erf(x):
    """
    function with parameters defined inside
    """
    # save the sign of x
    sign = 1
    if x < 0:
        sign = -1
    x = abs(x)

    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*math.exp(-x*x)
    return sign*y # erf(-x) = -erf(x)


def erfc(x):
    return 1-erf(x)


def model_gauss_peak(A, sigma, dx):
    """
    model a gaussian fluorescence peak 
    refer to van espen, spectrum evaluation in van grieken, 
    handbook of x-ray spectrometry, 2nd ed, page 182 ff
    
    Parameters:
    -----------
        A: float
              intensity of gaussian function
        sigma: float
               standard deviation
        x: array
           data in x coordinate, relative to center
    """
    
    counts = A / ( sigma *np.sqrt(2.*np.pi)) * np.exp( -0.5* ((dx / sigma)**2) )

    return counts



def model_gauss_step(A, sigma, dx, peak_E):
    """
    use scipy erfc function
    erfc = 1-erf
    refer to van espen, spectrum evaluation,
    in van grieken, handbook of x-ray spectrometry, 2nd ed, page 182
    
    Parameters:
    -----------
    A: float
       intensity or height
    sigma: float
           standard deviation
    dx: array
        data in x coordinate, x > 0
    peak_E: ???
        
    """
    
    counts = A / 2. /  peak_E * scipy.special.erfc(dx/(np.sqrt(2)*sigma))

    return counts



def model_gauss_tail(A, sigma, dx, gamma):
    """
    models a gaussian tail function
    refer to van espen, spectrum evaluation,
    in van grieken, handbook of x-ray spectrometry, 2nd ed, page 182
    
    Parameters:
    -----------
    A: float
        intensity or height
    sigma: float
            control peak width
    dx: array
        data in x coordinate
    gamma: float
           normalization factor
    """

    dx_neg = dx.copy()
    wo_neg = (np.nonzero(dx_neg > 0.))[0]
    if wo_neg.size > 0:
        dx_neg[wo_neg] = 0.
    temp_a = np.exp(dx_neg/ (gamma * sigma))
    counts = A / 2. / gamma / sigma / np.exp(-0.5/(gamma**2)) *  \
        temp_a * scipy.special.erfc( dx  /( np.sqrt(2)*sigma) + (1./(gamma*np.sqrt(2)) )  )

    return counts



def elastic_peak(coherent_sct_energy, fwhm_offset, fwhm_fanoprime, ev, A):
    """
    model elastic peak as a gaussian function
    
    Parameters:
    -----------
    coherent_sct_energy: float
                         incident energy                         
    fwhm_offset: float
                 global parameter for peak width    
    fwhm_fanoprime: float
                    global parameter for peak width
    ev: array
        energy value
    A: float:
       peak amplitude of gaussian peak
    
    Returns:
    --------
    value: array
           elastic peak
    sigma: float
           peak width
                     
    """
    # energy to create a hole-electron pair
    # for Ge 2.96, for Si 3.61 at 300K
    # needs to double check this value
    epsilon = 2.96
    temp_val = 2 * np.sqrt( 2 * np.log( 2 ) )
    sigma = np.sqrt( ( fwhm_offset / temp_val )**2  + \
                      coherent_sct_energy * epsilon*fwhm_fanoprime  )
    
    delta_energy = ev - coherent_sct_energy

    value = model_gauss_peak(A, sigma, delta_energy)
    
    return value, sigma




def compton_peak(coherent_sct_energy, fwhm_offset, fwhm_fanoprime, 
                 compton_angle, compton_fwhm_corr, compton_amplitude,
                 compton_f_step, compton_f_tail, compton_gamma,
                 compton_hi_f_tail, compton_hi_gamma,
                 ev, A, matrix = False):
    """
    model compton peak
    
    Parameters:
    ----------
    coherent_sct_energy: float
                         incident energy                         
    fwhm_offset: float
                 global parameter for peak width    
    fwhm_fanoprime: float
                    global parameter for peak width
    compton related parameters: float 
    compton_angle, compton_fwhm_corr, compton_amplitude,
    compton_f_step, compton_f_tail, compton_gamma,
    compton_hi_f_tail, compton_hi_gamma,
    
    ev: array
        energy value
    A: float
       peak height
    
    Returns:
    --------
    counts: array
            compton peak
    sigma: float
           related to gaussian peak width
    faktor: float
            normalization factor
    """
    compton_E = coherent_sct_energy / ( 1. + ( coherent_sct_energy / 511. ) * \
                                     ( 1. -np.cos( compton_angle * np.pi / 180. ) ) )
    
    # energy to create a hole-electron pair
    # for Ge 2.96, for Si 3.61 at 300K
    # needs to double check this value
    epsilon = 2.96
    temp_val = 2 * np.sqrt( 2 * np.log( 2 ) )
    sigma = np.sqrt( ( fwhm_offset / temp_val )**2 + compton_E * epsilon * fwhm_fanoprime  )
    
    #local_sigma = sigma*p[14]

    delta_energy = ev.copy() - compton_E

    counts = np.zeros(len(ev))

    faktor = 1. / (1. + compton_f_step + compton_f_tail + compton_hi_f_tail)
    
    if matrix == False :
        faktor = faktor * (10.**compton_amplitude)
        
    value = faktor * model_gauss_peak(A, sigma*compton_fwhm_corr, delta_energy)
    counts = counts + value

    # compton peak, step
    if compton_f_step > 0.:
        value = faktor * compton_f_step
        value = value * model_gauss_step(A, sigma, delta_energy, compton_E)
        counts = counts + value
    
    # compton peak, tail on the low side
    value = faktor * compton_f_tail
    value = value * model_gauss_tail(A, sigma, delta_energy, compton_gamma)
    counts = counts + value

    # compton peak, tail on the high side
    value = faktor * compton_hi_f_tail
    value = value * model_gauss_tail(A, sigma, -1.*delta_energy, compton_hi_gamma)
    counts = counts + value

    return counts, sigma, faktor
    



    
    

    