# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# @author: Li Li (lili@bnl.gov)                                        #
# created on 03/27/2015                                                #
#                                                                      #
# Original code from Xiaojing Huang (xjhuang@bnl.gov) and Li Li        #
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
import six
import numpy as np
import time

import logging
logger = logging.getLogger(__name__)


def _dist(dims):
    """
    Create array with pixel value equals to the distance from array center.

    Parameters
    ----------
    dims : list or tuple
        shape of array to create

    Returns
    -------
    arr : np.ndarray
        ND array whose pixels are equal to the distance from the center
        of the array of shape `dims`
    """
    dist_sum = []
    shape = np.ones(len(dims))
    for idx, d in enumerate(dims):
        vec = (np.arange(d) - d // 2) ** 2
        shape[idx] = -1
        vec = vec.reshape(*shape)
        shape[idx] = 1
        dist_sum.append(vec)

    return np.sqrt(np.sum(dist_sum, axis=0))


def gauss(dims, sigma):
    """
    Generate Gaussian function in 2D or 3D.

    Parameters
    ----------
    dims : list or tuple
        shape of the data
    sigma : float
        standard deviation of gaussian function

    Returns
    -------
    Array :
        ND gaussian
    """
    x = _dist(dims)
    y = np.exp(-(x / sigma)**2 / 2)
    return y / np.sum(y)


def convolution(array1, array2):
    """
    Calculate convolution of two arrays. Transfer into q space to perform the calculation.

    Parameters
    ----------
    array1 : array
        The size of array1 needs to be normalized.
    array2 : array
        The size of array2 keeps the same

    Returns
    -------
    array :
        convolution result

    Notes
    -----
    Another option is to use scipy.signal.fftconvolve. Some differences between
    the scipy function and this function were found at the boundary.  See
    `this issue on github <https://github.com/Nikea/scikit-xray/issues/258>`_
    for details.
    """
    fft_norm = lambda x:  np.fft.fftshift(np.fft.fftn(x)) / np.sqrt(np.size(x))
    fft_1 = fft_norm(array1)
    fft_2 = fft_norm(array2)
    return np.abs(np.fft.ifftshift(np.fft.ifftn(fft_1*fft_2)) *
                  np.sqrt(np.size(array2)))


def pi_modulus(array_in, diffraction_array, pi_modulus_flag):
    """
    Transfer sample from real space to q space.
    Use constraint based on diffraction pattern from experiments.

    Parameters
    ----------
    array_in : array
        reconstructed pattern in real space
    diffraction_array : array
        experimental data
    pi_modulus_flag : str
        Complex or Real

    Returns
    -------
    array :
        updated pattern in q space
    """
    thresh_v = 1e-12
    diff_tmp = np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(array_in)))
    index = np.where(diffraction_array > 0)
    diff_tmp[index] = diffraction_array[index] * diff_tmp[index] / (np.abs(diff_tmp[index]) + thresh_v)

    if pi_modulus_flag == 'Complex':
        return np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(diff_tmp)))
    if pi_modulus_flag == 'Real':
        return np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(diff_tmp))))


def find_support(sample_obj,
                 sw_sigma, sw_threshold):
    """
    Update sample area based on thresholds.

    Parameters
    ----------
    sample_obj : array
        sample for reconstruction
    sw_sigma : float
        sigma for gaussian in shrinkwrap method
    sw_threshold : float
        threshold used in shrinkwrap method

    Returns
    -------
    new_sample : array
        updated sample
    s_index : array
        index for sample area
    s_out_index : array
        index not for sample area
    """

    ga = np.abs(sample_obj)
    gauss_fun = gauss(sample_obj.shape, sw_sigma)
    gauss_fun = gauss_fun / np.max(gauss_fun)
    gb = convolution(ga, gauss_fun)
    gb_max = np.max(gb)
    s_index = np.where(gb >= sw_threshold*gb_max)
    s_out_index = np.where(gb < sw_threshold*gb_max)

    new_sample = np.zeros_like(sample_obj)
    new_sample[s_index] = 1.0
    #self.sup = sup.copy()
    #self.sup_index = s_index

    return new_sample, s_index, s_out_index


def pi_support(sample_obj, index_v):
    sample_obj = sample_obj.copy()
    sample_obj[index_v] = 0.0
    return sample_obj


class CDI(object):

    def __init__(self, diffamp, **kwargs):
        """
        Parameters
        ----------
        deffmap : array
            diffraction pattern from experiments
        kwargs : dict
            parameters related to cdi reconstruction
        """

        self.diff_array = np.array(diffamp)     # diffraction data

        self.ndim = self.diff_array.ndim    # 2D or 3D case
        if self.ndim == 2:
            self.nx, self.ny = np.shape(self.diff_array)    # array dimension
        if self.ndim == 3:
            self.nx, self.ny, self.nz = np.shape(self.diff_array)    # array dimension

        self.beta = kwargs['beta']  # feedback parameter for difference map algorithm, around 1.15
        self.start_ave = kwargs['start_ave']  # 0.8
        self.gamma_1 = -1/self.beta
        self.gamma_2 = 1/self.beta

        self.init_obj_flag = True
        self.init_sup_flag = True
        self.sup_radius = 150.
        self.sup_shape = 'Disk'  # 'Box' or 'Disk'

        self.pi_modulus_flag = kwargs['pi_modulus_flag']  # 'Complex'

        # parameters related to shrink wrap
        self.shrink_wrap_flag = kwargs['shrink_wrap_flag']
        self.sw_sigma = 0.5
        self.sw_threshold = 0.1
        self.sw_start = 0.2
        self.sw_end = 0.8
        self.sw_step = 10

    def get_sample_obj(self):
        pass

    def init_obj(self):
        """Initiate phase. Focus on 2D here.
        """
        pha_tmp = np.random.uniform(0, 2*np.pi, (self.nx, self.ny))
        self.obj = (np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(self.diff_array * np.exp(1j*pha_tmp))))
                    * np.sqrt(np.size(self.diff_array)))

    def init_sup(self):
        """ Initiate sample support.
        """
        self.sup = np.zeros_like(self.diff_array)
        if self.ndim == 2:
            if self.sup_shape == 'Box':
                self.sup[self.nx/2-self.sup_radius: self.nx/2+self.sup_radius,
                         self.ny/2-self.sup_radius:self.ny/2+self.sup_radius] = 1.0
        if self.ndim == 3:
            if self.sup_shape == 'Box':
                self.sup[self.nx/2-self.sup_radius:self.nx/2+self.sup_radius,
                         self.ny/2-self.sup_radius:self.ny/2+self.sup_radius,
                         self.nz/2-self.sup_radius:self.nz/2+self.sup_radius] = 1.0
        if self.sup_shape == 'Disk':
            dummy = _dist(np.shape(self.diff_array))
            self.sup_index = np.where(dummy <= self.sup_radius)
            self.sup[self.sup_index] = 1.0

    def recon(self, n_iterations=1000):
        """
        Run reconstruction.

        Parameters
        ---------
        n_iterations : int
            number of reconstructions to run.
        """

        # initiate shape and phase
        if(self.init_obj_flag):
            self.init_obj()
        if(self.init_sup_flag):
            self.init_sup()

        self.obj_error = np.zeros(n_iterations)
        self.diff_error = np.zeros(n_iterations)

        ave_i = 0
        self.time_start = time.time()
        for n in range(n_iterations):
            self.obj_old = self.obj.copy()

            self.obj_a = pi_modulus(self.obj, self.diff_array, self.pi_modulus_flag)
            self.obj_a = (1 + self.gamma_2) * self.obj_a - self.gamma_2 * self.obj
            self.obj_a = pi_support(self.obj_a, self.sup_index)

            self.obj_b = pi_support(self.obj, self.sup_index)
            self.obj_b = (1 + self.gamma_1) * self.obj_b - self.gamma_1 * self.obj
            self.obj_b = pi_modulus(self.obj_b, self.diff_array, self.pi_modulus_flag)

            self.obj = self.obj + self.beta * (self.obj_a - self.obj_b)

            # calculate errors
            #self.cal_obj_error(n)
            #self.cal_diff_error(n)

            if self.shrink_wrap_flag:
                if((n >= (self.sw_start * n_iterations)) and (n <= (self.sw_end * n_iterations))):
                    if np.mod(n, self.sw_step) == 0:
                        #self.sup_old = self.sup.copy()
                        print('refine support with shrinkwrap')
                        self.obj = find_support(self.obj, self.sw_sigma, self.sw_threshold)
                        #self.cal_error_sup()

            if n > int(self.start_ave*n_iterations):
                self.obj_ave += self.obj
                ave_i += 1

            print('{} object_chi= {}, diff_chi={}'.format(n, self.obj_error[n],
                                                          self.diff_error[n]))

        self.obj_ave = self.obj_ave / ave_i
        self.time_end = time.time()

        print('object size: {}'.format(np.shape(self.diff_array)))
        print('{} iterations takes {} sec'.format(n_iterations, self.time_end - self.time_start))

        return self.obj_ave