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
    arr : array
        ND gaussian
    """
    x = _dist(dims)
    y = np.exp(-(x / sigma)**2 / 2)
    return y / np.sum(y)


_fft_helper = lambda x: np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(x)))
_ifft_helper = lambda x: np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(x)))


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
    arr : array
        convolution result

    Notes
    -----
    Another option is to use scipy.signal.fftconvolve. Some differences between
    the scipy function and this function were found at the boundary.  See
    `this issue on github <https://github.com/Nikea/scikit-xray/issues/258>`_
    for details.
    """
    fft_1 = _fft_helper(array1) / np.sqrt(np.size(array1))
    fft_2 = _fft_helper(array2) / np.sqrt(np.size(array2))
    return np.abs(_ifft_helper(fft_1*fft_2) * np.sqrt(np.size(array2)))


def pi_modulus(array_in, diff_array,
               thresh_v=1e-12):
    """
    Transfer sample from real space to q space.
    Use constraint based on diffraction pattern from experiments.

    Parameters
    ----------
    array_in : array
        reconstructed pattern in real space
    diff_array : array
        experimental data
    thresh_v : float
        add small value to avoid the case of dividing something by zero

    Returns
    -------
    arr : array
        updated pattern in q space
    """
    diff_tmp = _fft_helper(array_in) / np.sqrt(np.size(array_in))
    index = np.where(diff_array > 0)
    diff_tmp[index] = diff_array[index] * diff_tmp[index] / (np.abs(diff_tmp[index]) + thresh_v)

    return _ifft_helper(diff_tmp) * np.sqrt(np.size(diff_array))


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
    new_sup : array
        updated sample support
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

    new_sup = np.zeros_like(sample_obj)
    new_sup[s_index] = 1

    return new_sup, s_index, s_out_index


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
    sample_obj : array
        sample object with proper cut.
    """
    sample_obj = np.array(sample_obj)
    sample_obj[index_v] = 0
    return sample_obj


def cal_relative_error(x_old, x_new):
    """
    Relative error is calculated as the ratio of the difference between the new and
    the original arrays to the norm of the original array.

    Parameters
    ----------
    x_old : array
        previous data set
    x_new : array
        new data set

    Returns
    -------
    float :
        relative error
    """
    return np.linalg.norm(x_new - x_old)/np.linalg.norm(x_old)


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
    new_diff = np.abs(_fft_helper(sample_obj)) / np.sqrt(np.size(sample_obj))
    return cal_relative_error(diff_array, new_diff)


def generate_random_phase_field(diff_array):
    """
    Initiate random phase.

    Parameters
    ----------
    diff_array : array
        diffraction pattern

    Returns
    -------
    sample_obj : array
        sample information with phase
    """
    pha_tmp = np.random.uniform(0, 2*np.pi, diff_array.shape)
    sample_obj = (_ifft_helper(diff_array * np.exp(1j*pha_tmp))
                  * np.sqrt(np.size(diff_array)))
    return sample_obj


def generate_box_support(sup_radius, shape_v):
    """
    Generate support area as a box for either 2D or 3D cases.

    Parameters
    ----------
    sup_radius : float
        radius of support
    shape_v : list
        shape of diffraction pattern, which can be either 2D or 3D case.

    Returns
    -------
    sup : array
        support with a box area
    """
    sup = np.zeros(shape_v)
    if len(shape_v) == 2:
        nx, ny = shape_v
        sup[nx/2-sup_radius: nx/2+sup_radius,
            ny/2-sup_radius: ny/2+sup_radius] = 1
    elif len(shape_v) == 3:
        nx, ny, nz = shape_v
        sup[nx/2-sup_radius: nx/2+sup_radius,
            ny/2-sup_radius: ny/2+sup_radius,
            nz/2-sup_radius: nz/2+sup_radius] = 1
    return sup


def generate_disk_support(sup_radius, shape_v):
    """
    Generate support area as a disk for either 2D or 3D cases.

    Parameters
    ----------
    sup_radius : float
        radius of support
    shape_v : list
        shape of diffraction pattern, which can be either 2D or 3D case.

    Returns
    -------
    sup : array
        support with a disk area
    """
    sup = np.zeros(shape_v)
    dummy = _dist(shape_v)
    sup_index = np.where(dummy <= sup_radius)
    sup[sup_index] = 1
    return sup


def cdi_recon(diff_array, sample_obj, sup,
              beta=1.15, start_ave=0.8, pi_modulus_flag='Complex',
              sw_flag=True, sw_sigma=0.5, sw_threshold=0.1, sw_start=0.2,
              sw_end=0.8, sw_step=10, n_iterations=1000):
    """
    Run reconstruction with difference map algorithm.

    Parameters
    ---------
    diff_array : array
        diffraction pattern from experiments
    sample_obj : array
        initial sample with phase
    sup : array
        initial support
    beta : float, optional
        feedback parameter for difference map algorithm.
        default is 1.15.
    start_ave : float, optional
        define the point to start doing average.
        default is 0.8.
    pi_modulus_flag : str, optional
        'Complex' or 'Real', defining the way to perform pi_modulus calculation.
        default is 'Complex'.
    sw_flag : Bool, optional
        flag to use shrinkwrap algorithm or not.
        default is True.
    sw_sigma : float, optional
        gaussian width used in sw algorithm.
        default is 0.5.
    sw_threshold : float, optional
        shreshold cut in sw algorithm.
        default is 0.1.
    sw_start : float, optional
        at which point to start to do shrinkwrap.
        defualt is 0.2
    sw_end : float, optional
        at which point to stop shrinkwrap.
        defualt is 0.8
    sw_step : float, optional
        the frequency to perform sw algorithm.
        defualt is 10
    n_iterations : int, optional
        number of iterations to run.
        default is 1000.

    Returns
    -------
    obj_ave : array
        reconstructed sample object
    error_dict : dict
        Error information for all iterations. The dict keys include
        obj_error, diff_error and sup_error. Obj_error is a list of
        the relative error of sample object. Diff_error is calculated as
        the difference between new diffraction pattern and the original
        diffraction pattern. And sup_error stores the size of the
        sample support.
    """

    diff_array = np.array(diff_array)     # diffraction data

    gamma_1 = -1/beta
    gamma_2 = 1/beta

    # get support index
    sup_index = np.where(sup == 1)
    sup_out_index = np.where(sup != 1)

    error_dict = {}
    obj_error = np.zeros(n_iterations)
    diff_error = np.zeros(n_iterations)
    sup_error = np.zeros(n_iterations)

    sup_old = np.zeros_like(diff_array)
    obj_ave = np.zeros_like(diff_array).astype(complex)
    ave_i = 0

    time_start = time.time()
    for n in range(n_iterations):
        obj_old = np.array(sample_obj)

        obj_a = pi_modulus(sample_obj, diff_array)
        if pi_modulus_flag.lower() == 'real':
            obj_a = np.abs(obj_a)

        obj_a = (1 + gamma_2) * obj_a - gamma_2 * sample_obj
        obj_a = pi_support(obj_a, sup_out_index)

        obj_b = pi_support(sample_obj, sup_out_index)
        obj_b = (1 + gamma_1) * obj_b - gamma_1 * sample_obj

        obj_b = pi_modulus(obj_b, diff_array)
        if pi_modulus_flag.lower() == 'real':
            obj_b = np.abs(obj_b)

        sample_obj += beta * (obj_a - obj_b)

        # calculate errors
        obj_error[n] = cal_relative_error(obj_old, sample_obj)
        diff_error[n] = cal_diff_error(sample_obj, diff_array)

        if sw_flag is True:
            if((n >= (sw_start * n_iterations)) and (n <= (sw_end * n_iterations))):
                if np.mod(n, sw_step) == 0:
                    logger.info('Refine support with shrinkwrap')
                    sup, sup_index, sup_out_index = find_support(sample_obj,
                                                                 sw_sigma,
                                                                 sw_threshold)
                    sup_error[n] = np.sum(sup_old)
                    sup_old = np.array(sup)

        if n > start_ave*n_iterations:
            obj_ave += sample_obj
            ave_i += 1

        logger.info('{} object_chi= {}, diff_chi={}'.format(n, obj_error[n],
                                                            diff_error[n]))

    obj_ave = obj_ave / ave_i
    time_end = time.time()

    logger.info('object size: {}'.format(np.shape(diff_array)))
    logger.info('{} iterations takes {} sec'.format(n_iterations,
                                                    time_end - time_start))

    error_dict['obj_error'] = obj_error
    error_dict['diff_error'] = diff_error
    error_dict['sup_error'] = sup_error

    return obj_ave, error_dict
