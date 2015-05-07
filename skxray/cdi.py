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
from scipy import signal

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
    #return np.abs(np.fft.ifftshift(np.fft.ifftn(fft_1*fft_2)) *
    #              np.sqrt(np.size(array2)))
    return signal.fftconvolve(array1, array2, mode='same')


def pi_modulus(array_in, diff_array, pi_modulus_flag):
    """
    Transfer sample from real space to q space.
    Use constraint based on diffraction pattern from experiments.

    Parameters
    ----------
    array_in : array
        reconstructed pattern in real space
    diff_array : array
        experimental data
    pi_modulus_flag : str
        Complex or Real

    Returns
    -------
    array :
        updated pattern in q space
    """
    thresh_v = 1e-12
    diff_tmp = np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(array_in))) / np.sqrt(np.size(array_in))
    index = np.where(diff_array > 0)
    diff_tmp[index] = diff_array[index] * diff_tmp[index] / (np.abs(diff_tmp[index]) + thresh_v)

    if pi_modulus_flag == 'Complex':
        return np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(diff_tmp))) * np.sqrt(np.size(diff_array))
    elif pi_modulus_flag == 'Real':
        return np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(diff_tmp)))) * np.sqrt(np.size(diff_array))


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

    sample_obj = np.abs(sample_obj)
    gauss_fun = gauss(sample_obj.shape, sw_sigma)
    gauss_fun = gauss_fun / np.max(gauss_fun)
    conv_fun = convolution(sample_obj, gauss_fun)
    conv_max = np.max(conv_fun)

    s_index = np.where(conv_fun >= (sw_threshold*conv_max))
    s_out_index = np.where(conv_fun < (sw_threshold*conv_max))

    new_sample = np.zeros_like(sample_obj)
    new_sample[s_index] = 1.0
    #self.sup = sup.copy()
    #self.sup_index = s_index

    return new_sample, s_index, s_out_index


def pi_support(sample_obj, index_v):
    """
    Define sample shape by cutting unnecessary values.

    Parameters
    ----------
    sample_obj : array
        sample data
    index_v : array
        index to define sample area

    Returns
    -------
    array :
        sample object with proper cut.
    """
    sample_obj = np.array(sample_obj)
    sample_obj[index_v] = 0.0
    return sample_obj


def cal_sample_error(sample_old, sample_new):
    """
    Calculate error in sample space.

    Parameters
    ----------
    sample_old : array
        old sample data
    sample_new : array
        new sample data

    Returns
    -------
    float :
        relative error
    """
    sample_error = (np.sqrt(np.sum((np.abs(sample_new - sample_old))**2)) /
                    np.sqrt(np.sum((np.abs(sample_old))**2)))
    return sample_error


def cal_diff_error(sample_obj, diff_array):
    """
    Calculate the error in q space.

    Parameters
    ----------
    sample_obj : array
        sample data
    diff_array : array
        diffraction pattern

    Returns
    -------
    float :
        relative error in q space
    """

    fft_norm = np.abs(np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(sample_obj))) /
                      np.sqrt(np.size(sample_obj)))
    diff_error = (np.sqrt(np.sum((np.abs(fft_norm - diff_array))**2)) /
                  np.sqrt(np.sum((np.abs(diff_array))**2)))
    return diff_error


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

        self.pi_modulus_flag = kwargs['pi_modulus_flag']  # 'Complex'

        self.beta = kwargs['beta']  # feedback parameter for difference map algorithm, around 1.15
        self.start_ave = kwargs['start_ave']  # 0.8

        # initial support
        self.init_obj_flag = kwargs['init_obj_flag']
        self.init_sup_flag = kwargs['init_sup_flag']
        self.sup_radius = kwargs['support_radius']  # 150
        self.sup_shape = kwargs['support_shape']  # 'Box' or 'Disk'

        # parameters related to shrink wrap
        self.shrink_wrap_flag = kwargs['shrink_wrap_flag']
        self.sw_sigma = kwargs['sw_sigma']  # 0.5
        self.sw_threshold = kwargs['sw_threshold']  # 0.1
        self.sw_start = kwargs['sw_start']  # 0.2
        self.sw_end = kwargs['sw_end']  # 0.8
        self.sw_step = kwargs['sw_step']  # 10

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
                         self.ny/2-self.sup_radius:self.ny/2+self.sup_radius] = 1
        if self.ndim == 3:
            if self.sup_shape == 'Box':
                self.sup[self.nx/2-self.sup_radius:self.nx/2+self.sup_radius,
                         self.ny/2-self.sup_radius:self.ny/2+self.sup_radius,
                         self.nz/2-self.sup_radius:self.nz/2+self.sup_radius] = 1
        if self.sup_shape == 'Disk':
            dummy = _dist(np.shape(self.diff_array))
            self.sup_index = np.where(dummy <= self.sup_radius)
            self.sup[self.sup_index] = 1

        self.sup_index = np.where(self.sup == 1)
        self.sup_out_index = np.where(self.sup == 0)

    def recon(self, n_iterations=1000):
        """
        Run reconstruction with difference map algorithm.

        Parameters
        ---------
        n_iterations : int
            number of reconstructions to run

        Returns
        -------
        array :
            reconstructed sample object
        """
        gamma_1 = -1/self.beta
        gamma_2 = 1/self.beta

        # initiate shape and phase
        if(self.init_obj_flag):
            self.init_obj()
        if(self.init_sup_flag):
            self.init_sup()

        self.obj_error = np.zeros(n_iterations)
        self.diff_error = np.zeros(n_iterations)

        obj_ave = np.zeros_like(self.obj)
        ave_i = 0

        self.time_start = time.time()
        for n in range(n_iterations):
            self.obj_old = np.array(self.obj)

            obj_a = pi_modulus(self.obj, self.diff_array, self.pi_modulus_flag)
            obj_a = (1 + gamma_2) * obj_a - gamma_2 * self.obj
            obj_a = pi_support(obj_a, self.sup_out_index)

            obj_b = pi_support(self.obj, self.sup_out_index)
            obj_b = (1 + gamma_1) * obj_b - gamma_1 * self.obj
            obj_b = pi_modulus(obj_b, self.diff_array, self.pi_modulus_flag)

            self.obj = self.obj + self.beta * (obj_a - obj_b)

            # calculate errors
            self.obj_error[n] = cal_sample_error(self.obj_old, self.obj)
            self.diff_error[n] = cal_diff_error(self.obj, self.diff_array)
            #self.cal_diff_error(n)

            if self.shrink_wrap_flag is True:
                if((n >= (self.sw_start * n_iterations)) and (n <= (self.sw_end * n_iterations))):
                    if np.mod(n, self.sw_step) == 0:
                        #self.sup_old = self.sup.copy()
                        #logger.info('refine support with shrinkwrap')
                        print('refine support with shrinkwrap')
                        self.obj, self.sup_index, self.sup_out_index = find_support(self.obj,
                                                                                    self.sw_sigma,
                                                                                    self.sw_threshold)
                        #self.cal_error_sup()

            if n > self.start_ave*n_iterations:
                obj_ave += self.obj
                ave_i += 1

            print('{} object_chi= {}, diff_chi={}'.format(n, self.obj_error[n],
                                                          self.diff_error[n]))

        obj_ave = obj_ave / ave_i
        self.time_end = time.time()

        logger.info('object size: {}'.format(np.shape(self.diff_array)))
        logger.info('{} iterations takes {} sec'.format(n_iterations, self.time_end - self.time_start))

        return obj_ave
