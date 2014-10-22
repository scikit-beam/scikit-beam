#!/usr/bin/env python
# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
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
This is a unit/integrated testing script for dpc.py, which conducts 
Differential Phase Contrast (DPC) imaging based on Fourier-shift fitting.
    
DPC calculation steps
---------------------
1. Set parameters
2. Load the reference image
3. Dimension reduction along x and y direction
4. 1-D IFFT
5. Same calculation on each diffraction pattern
    5.1. Read a diffraction pattern
    5.2. Dimension reduction along x and y direction
    5.3. 1-D IFFT
    5.4. Nonlinear fitting
6. Reconstruct the final phase image
7. Save intermediate and final results
    
"""
  
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
                            
import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)

import dpc


def test_image_reduction_default():
    """
    Test image reduction when default parameters (roi and bad_pixels) are used 
    
    """
    
    # Generate a 2D matrix
    img = np.arange(100).reshape(10, 10)
    
    # Expected results
    xsum = [450, 460, 470, 480, 490, 500, 510, 520, 530, 540]
    ysum = [45, 145, 245, 345, 445, 545, 645, 745, 845, 945]
    
    # call image reduction
    xline, yline = dpc.image_reduction(img)
    
    assert_array_equal(xline, xsum)
    
    assert_array_equal(yline, ysum)
    

def test_image_reduction():
    """
    Test image reduction when the following parameters are used:
    roi = (3, 3, 8, 8);
    bad_pixels = [(0, 1), (4, 4), (7, 8)]
    
    """

    # generate a 2D matrix
    img = np.arange(100).reshape(10, 10)    
    
    # set up roi and bad_pixels
    roi = (3, 3, 8, 8)
    bad_pixels = [(0, 1), (4, 4), (7, 8)]
    
    # Expected results
    xsum = [265, 226, 275, 280, 285]
    ysum = [175, 181, 275, 325, 375]
    
    # call image reduction
    xline, yline = dpc.image_reduction(img, roi, bad_pixels)    
    
    assert_array_equal(xline, xsum)
    
    assert_array_equal(yline, ysum)
        

def test_ifft1D_shift():
    """
    Test 1D shifted IFFT
    
    """
    
    input = np.arange(10)
    
    output = dpc.ifft1D_shift(input)
    
    expected = [-0.5 + 0.j, -0.5 + 0.16245985j, -0.5 + 0.36327126j, 
                -0.5 + 0.68819096j, -0.5 + 1.53884177j, 4.5 + 0.j, 
                -0.5 - 1.53884177j, -0.5 - 0.68819096j, -0.5 - 0.36327126j, 
                -0.5 - 0.16245985j]
    
    assert_array_almost_equal(output, expected)
    
    
def test_rss():
    """
    Test _rss
    
    """
    
    xdata = np.arange(10)
    ydata = [0, 1.68770792 + 1.07314584j, -3.64452105 - 1.64847394j,
             5.76102172 + 1.67649299j, -7.91993997 - 1.12896006j, 
             10.00000000 + 0.j, -11.87990996 + 1.6934401j, 
             13.44238401 - 3.91181697j, -14.57808419 + 6.59389576j, 
             15.18937126 - 9.65831252j]
    v = [2, 3]
    
    residue = dpc._rss(v, xdata, ydata)
    
    assert_almost_equal(residue, 0)
    

def test_dpc_fit():
    """
    Test dpc_fit
    
    """
    
    # Test 1 (succeeded): res = [1.34, 0.23]
    xdata = np.arange(10)
    ydata = [0, 0.81179901 - 1.06610617j, 2.06693932 - 1.70591965j,
             3.60213104 - 1.78467139j, 5.21885188 - 1.22195953j, 
             6.70000000 + 0.j, 7.82827782 + 1.83293929j, 
             8.40497243 + 4.16423324j, 8.26775728 + 6.82367859j, 
             7.30619109 + 9.59495554j]
    res = dpc.dpc_fit(xdata, ydata)
    assert_array_almost_equal(res, [1.34, 0.23])
    
    # Test 2 (succeeded): res = [0.88, 0.28]
    ydata = [0, 0.38340055 - 0.79208839j, 1.17473457 - 1.31057189j,
             2.23675349 - 1.40233156j,  3.38291514 - 0.97277188j, 
             4.40000000 + 0.j, 5.07437271 + 1.45915782j, 
             5.21909148 + 3.27210698j, 4.69893829 + 5.24228756j,
             3.45060497 + 7.1287955j]
    res = dpc.dpc_fit(xdata, ydata)
    assert_array_almost_equal(res, [0.88, 0.28]) 
    
    """
    # Test 3 (failed): res = [-0.25595591, -0.27603199]
    ydata = [0, -0.71170192 - 0.31918705j, -0.70539481 - 1.3914087j,
             0.48961848 - 2.28820317j, 2.42602688 - 1.96183423j, 3.9,
             3.63904032 + 2.94275135j, 1.14244312 + 5.33914073j, 
             -2.82157925 + 5.56563478j, -6.40531730+2.87268347j]
    res = dpc.dpc_fit(xdata, ydata)
    assert_array_almost_equal(res, [0.78, 0.68])    
    """    
   

def test_dpc_end_to_end():
    """
    Integrated test for DPC
    
    """
    
    # 1. Set parameters
    
    start_point=[1, 0]
    pixel_size=55
    focus_to_det=1.46e6
    rows = 32
    cols = 32
    energy=19.5
    roi=None
    pad=1
    w=1.
    bad_pixels=None
    solver='Nelder-Mead'
    img_size = (40, 40)

    # Initialize a, gx, gy and phi
    a = np.zeros((rows, cols), dtype='d')
    gx = np.zeros((rows, cols), dtype='d')
    gy = np.zeros((rows, cols), dtype='d')
    phi = np.zeros((rows, cols), dtype='d')

    # 2. Load the reference image
    ref = np.ones(img_size)

    # 3. Dimension reduction along x and y direction
    refx, refy = dpc.image_reduction(ref, roi=roi)

    # 4. 1-D IFFT
    ref_fx = np.fft.fftshift(np.fft.ifft(refx))
    ref_fy = np.fft.fftshift(np.fft.ifft(refy))

    # 5. Same calculation on each diffraction pattern
    for i in range(rows):
        print(i)
        for j in range(cols):
                
            try:
                # 5.1. Read a diffraction pattern
                im = np.ones(img_size)
                   
                # 5.2. Dimension reduction along x and y direction
                imx, imy = dpc.image_reduction(im, roi=roi)
            
                # 5.3. 1-D IFFT
                fx = np.fft.fftshift(np.fft.ifft(imx))
                fy = np.fft.fftshift(np.fft.ifft(imy))
                
                # 5.4. Nonlinear fitting
                _a, _gx = dpc.dpc_fit(ref_fx, fx)
                _a, _gy = dpc.dpc_fit(ref_fy, fy)
                            
                # Store one-point intermediate results
                gx[i, j] = _gx
                gy[i, j] = _gy
                a[i, j] = _a
        
            except Exception as ex:
                print('Failed to calculate %s: %s' % (filename, ex))
                gx[i, j] = 0
                gy[i, j] = 0
                a[i, j] = 0
    
    # Scale gx and gy. Not necessary all the time
    lambda_ = 12.4e-4 / energy
    gx *= - len(ref_fx) * pixel_size / (lambda_ * focus_to_det)
    gy *= len(ref_fy) * pixel_size / (lambda_ * focus_to_det)

    # 6. Reconstruct the final phase image
    phi = dpc.recon(gx, gy)
    
    assert_array_almost_equal(phi, np.zeros((rows, cols)))



if __name__ == "__main__":

    test_image_reduction_default()
    test_image_reduction()
    test_ifft1D_shift()
    test_rss()
    test_dpc_fit()
    
    test_dpc_end_to_end()
