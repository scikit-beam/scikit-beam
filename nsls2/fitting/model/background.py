'''
Copyright (c) 2014, Brookhaven National Laboratory
All rights reserved.

# @author: Li Li (lili@bnl.gov)
# created on 07/16/2014

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

import scipy.signal
import numpy as np
import matplotlib.pyplot as plt


def snip_method(spectrum, 
                e_off, e_lin, e_quad, 
                xmin=0, xmax=2048,
                epsilon = 2.96,
                spectral_binning=None, width=0.5):
    """
    use snip algorithm to obtain background

    Parameters:
    -----------
    spectrum : 1D array
        intensity spectrum
    e_off : float
        energy calibration, such as e_off + e_lin * energy + e_quad * energy^2
    e_lin : float
        energy calibration, such as e_off + e_lin * energy + e_quad * energy^2
    e_quad : float
        energy calibration, such as e_off + e_lin * energy + e_quad * energy^2       
    xmin : float
        smallest index to define the range
    xmax : float
        largest index to define the range
    epsilon: float
        energy to create a hole-electron pair 
        for Ge 2.96, for Si 3.61 at 300K
        needs to double check this value
    spectral_binning: int
        bin the data into different size
    width : int
        step size to shift background            

    Returns:
    --------
    background : 1D array
        output results with peak removed
    """

    background = np.array(spectrum)
    n_background = background.size

    energy = np.arange(n_background, dtype=np.float)
    
    if spectral_binning > 0:
        energy = energy * spectral_binning

    energy = e_off + energy * e_lin + energy**2 * e_quad
    
    temp_val = 2 * np.sqrt(2 * np.log(2))
    tmp = (e_off / temp_val)**2 + energy * epsilon * e_lin

    tmp[tmp < 0] = 0

    # transfer from std to fwhm
    fwhm = 2.35 * np.sqrt(tmp)

    #smooth the background
    if spectral_binning > 0 :
        s = scipy.signal.boxcar(3)
    else :
        s = scipy.signal.boxcar(5)
    
    # For background remove, we only care about the central parts 
    # where there are peaks. On the boundary part, we don't care 
    # the accuracy so much. But we need to pay attention to edge 
    # effects in general convolution.
    A = s.sum()
    background = scipy.signal.convolve(background,s,mode='same')/A


    # SNIP PARAMETERS
    window_rf = np.sqrt(2)

    window_p = width * fwhm / e_lin
    if spectral_binning > 0:
        window_p = window_p/2.

    background = np.log(np.log(background + 1) + 1)

    index = np.arange(n_background)

    #FIRST SNIPPING

    if spectral_binning > 0:
        no_iterations = 3
    else:
        no_iterations = 2

    for j in range(no_iterations):
        lo_index = np.clip(index - window_p, np.max([xmin, 0]), np.min([xmax, n_background - 1]))
        hi_index = np.clip(index + window_p, np.max([xmin, 0]), np.min([xmax, n_background - 1]))

        temp = (background[lo_index.astype(np.int)] + background[hi_index.astype(np.int)]) / 2.
        wo = np.where(background > temp)
        background[wo] = temp[wo]


    if spectral_binning > 0:
        no_iterations = 7
    else:
        no_iterations = 12

    current_width = window_p
    max_current_width = np.amax(current_width)

    while max_current_width >= 0.5:
        lo_index = index - current_width
        wo = np.where(lo_index < max((xmin, 0)))
        lo_index[wo] = max((xmin, 0))
        hi_index = index + current_width
        wo = np.where(hi_index > min((xmax, n_background-1)))
        hi_index[wo] = min((xmax, n_background-1))

        temp = (background[lo_index.astype(np.int)] + background[hi_index.astype(np.int)]) / 2.
        wo = np.where(background > temp)
        background[wo] = temp[wo]

        # decrease the width and repeat
        current_width = current_width / window_rf
        max_current_width = np.amax(current_width)

    background = np.exp( np.exp(background)-1.)-1.

    wo = np.where(np.isfinite(background) == False)
    background[wo] = 0.

    return background




def test():
    """
    test of background removal
    """
    data = np.loadtxt('../test_data.txt')
    data = data[0:1250]
    x = np.arange(len(data))

    e_list = [0.1, 0.01, 0]
    xmin = 0
    xmax = 1800
    bg = snip_method(data, 
                     e_list[0], e_list[1], e_list[2], 
                     xmin=xmin, xmax=xmax,
                     spectral_binning=0, width=0.5)

    #plt.plot(bg)



    plt.semilogy(x, data, x, bg)
    plt.show()
    return


if __name__=="__main__":
    test()

