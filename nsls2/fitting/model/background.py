# Copyright (c) Brookhaven National Lab 2O14
# All rights reserved
# BSD License
# See LICENSE for full text
# @author:  Li Li (lili@bnl.gov)
# created on 07/16/2014

# Original code: 
# Copyright (c) 2013, Stefan Vogt, Mirna Lerotic, Argonne National Laboratory All rights reserved.


import scipy.signal
import numpy as np
import matplotlib.pyplot as plt


def snip_method(spectrum, e_list, xmin, xmax,
                spectral_binning=0, width=0.5):
    """
    use snip algorithm to obtain background
    
    Parameters:
    -----------
        spectrum: 1D array
                  intensity spectrum
        e_list: 1D array
                [e_off, e_lin, e_quad] for calibration in position
        spectral_binning: int
                          bin the data into different size
        width: int
               step size for snip algorithm
    
    Returns:
    --------
        background: 1D array
                    with peak removed
    """

    background = spectrum.copy()
    n_background = background.size

    e_off = e_list[0]
    e_lin = e_list[1]
    e_quad = e_list[2]


    energy = np.arange(np.float(n_background))
    if spectral_binning > 0:
        energy = energy * spectral_binning

    energy = e_off + energy * e_lin + np.power(energy,2) * e_quad

    # energy to create a hole-electron pair
    # for Ge 2.96, for Si 3.61 at 300K
    # needs to double check this value
    epsilon = 2.96
    temp_val = 2 * np.sqrt( 2 * np.log( 2 ) )
    tmp = ( e_off / temp_val )**2 + energy * epsilon * e_lin
    
    wind = np.nonzero(tmp < 0)[0]
    tmp[wind] = 0.
    fwhm = 2.35 * np.sqrt(tmp)

    #original_bcgrd = background.copy()

    #smooth the background
    if spectral_binning > 0 :
        s = scipy.signal.boxcar(3)
    else :
        s = scipy.signal.boxcar(5)
    A = s.sum()
    background = scipy.signal.convolve(background,s,mode='same')/A


    # SNIP PARAMETERS
    window_rf = np.sqrt(2)

    window_p = width * fwhm / e_lin
    if spectral_binning > 0:
        window_p = window_p/2.

    background = np.log(np.log(background+1.)+1.)

    index = np.arange(np.float(n_background))
    
    #FIRST SNIPPING

    if spectral_binning > 0:
        no_iterations = 3
    else:
        no_iterations = 2

    for j in range(no_iterations):
        lo_index = index - window_p
        wo = np.where(lo_index < max((xmin, 0)))
        lo_index[wo] = max((xmin, 0))
        hi_index = index + window_p
        wo = np.where(hi_index > min((xmax, n_background-1)))
        hi_index[wo] = min((xmax, n_background-1))

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

        current_width = current_width / window_rf
        max_current_width = np.amax(current_width)


    background = np.exp(np.exp(background)-1.)-1.

    wo = np.where(np.isfinite(background) == False)
    background[wo] = 0.

    return background




def test():
    """
    test of background removal
    """
    data = np.loadtxt('test_data.txt')
    data = data[0:1100]
    x = np.arange(len(data))
    
    e_list = [0.1, 0.01, 0]
    xmin = 0
    xmax = 1000
    bg = snip_method(data, e_list, xmin, xmax,
                     spectral_binning=0, width=0.5)
    
    #plt.plot(bg)
    
    
    
    plt.semilogy(x, data, x, bg)
    plt.show()
    return


if __name__=="__main__":
    test()
    
