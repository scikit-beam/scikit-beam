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
from __future__ import absolute_import, division, print_function
import logging

import numpy as np
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_raises

import skbeam.core.utils as utils
from skbeam.core.correlation import (multi_tau_auto_corr,
                                     auto_corr_scat_factor,
                                     lazy_one_time,
                                     lazy_two_time, two_time_corr,
                                     two_time_state_to_results,
                                     one_time_from_two_time)
from skbeam.core.mask import bad_to_nan_gen


logger = logging.getLogger(__name__)


def setup():
    global num_levels, num_bufs, xdim, ydim, stack_size, img_stack, rois
    num_levels = 6
    num_bufs = 4  # must be even
    xdim = 256
    ydim = 512
    stack_size = 100
    img_stack = np.random.randint(1, 3, (stack_size, xdim, ydim))
    rois = np.zeros_like(img_stack[0])
    # make sure that the ROIs can be any integers greater than 1.
    # They do not have to start at 1 and be continuous
    rois[0:xdim//10, 0:ydim//10] = 5
    rois[xdim//10:xdim//5, ydim//10:ydim//5] = 3


def test_lazy_vs_original():
    setup()
    # run the correlation on the full stack
    full_gen_one = lazy_one_time(
        img_stack, num_levels, num_bufs, rois)
    for gen_state_one in full_gen_one:
        pass

    g2, lag_steps = multi_tau_auto_corr(num_levels, num_bufs,
                                        rois, img_stack)

    assert np.all(g2 == gen_state_one.g2)
    assert np.all(lag_steps == gen_state_one.lag_steps)

    full_gen_two = lazy_two_time(rois, img_stack, stack_size,
                                 num_bufs, num_levels)
    for gen_state_two in full_gen_two:
        pass
    final_gen_result_two = two_time_state_to_results(gen_state_two)

    two_time = two_time_corr(rois, img_stack, stack_size,
                             num_bufs, num_levels)

    assert np.all(two_time[0] == final_gen_result_two.g2)
    assert np.all(two_time[1] == final_gen_result_two.lag_steps)


def test_lazy_two_time():
    setup()
    # run the correlation on the full stack
    full_gen = lazy_two_time(rois, img_stack, stack_size,
                             stack_size, 1)
    for full_state in full_gen:
        pass
    final_result = two_time_state_to_results(full_state)

    # make sure we have essentially zero correlation in the images,
    # since they are random integers
    assert np.average(final_result.g2-1) < 0.01

    # run the correlation on the first half
    gen_first_half = lazy_two_time(rois, img_stack[:stack_size//2], stack_size,
                                   num_bufs=stack_size, num_levels=1)
    for first_half_state in gen_first_half:
        pass
    # run the correlation on the second half by passing in the state from the
    # first half
    gen_second_half = lazy_two_time(rois, img_stack[stack_size//2:],
                                    stack_size, num_bufs=stack_size,
                                    num_levels=1,
                                    two_time_internal_state=first_half_state)

    for second_half_state in gen_second_half:
        pass
    result = two_time_state_to_results(second_half_state)

    assert np.all(full_state.g2 == result.g2)


def test_lazy_one_time():
    setup()
    # run the correlation on the full stack
    full_gen = lazy_one_time(img_stack, num_levels, num_bufs, rois)
    for full_result in full_gen:
        pass

    # make sure we have essentially zero correlation in the images,
    # since they are random integers
    assert np.average(full_result.g2-1) < 0.01

    # run the correlation on the first half
    gen_first_half = lazy_one_time(
        img_stack[:stack_size//2], num_levels, num_bufs, rois)
    for first_half_result in gen_first_half:
        pass
    # run the correlation on the second half by passing in the state from the
    # first half
    gen_second_half = lazy_one_time(
        img_stack[stack_size//2:], num_levels, num_bufs, rois,
        internal_state=first_half_result.internal_state
    )

    for second_half_result in gen_second_half:
        pass

    assert np.all(full_result.g2 ==
                  second_half_result.g2)


def test_two_time_corr():
    setup()
    y = []
    for i in range(50):
        y.append(img_stack[0])
    two_time = two_time_corr(rois, np.asarray(y), 50,
                             num_bufs=50, num_levels=1)
    assert np.all(two_time[0])

    # check the number of buffers are even
    assert_raises(ValueError, two_time_corr, rois, np.asarray(y), 50,
                  num_bufs=25, num_levels=1)


def test_auto_corr_scat_factor():
    num_levels, num_bufs = 3, 4
    tot_channels, lags, dict_lags = utils.multi_tau_lags(num_levels, num_bufs)
    beta = 0.5
    relaxation_rate = 10.0
    baseline = 1.0

    g2 = auto_corr_scat_factor(lags, beta, relaxation_rate, baseline)

    assert_array_almost_equal(g2, np.array([1.5, 1.0, 1.0, 1.0, 1.0,
                                            1.0, 1.0, 1.0]), decimal=8)


def test_bad_images():
    setup()
    g2, lag_steps = multi_tau_auto_corr(4, num_bufs,
                                        rois, img_stack)
    # introduce bad images
    bad_img_list = [3, 21, 35, 48]
    # convert each bad image to np.nan array
    images = bad_to_nan_gen(img_stack, bad_img_list)

    # then use new images (including bad images)
    g2_n, lag_steps_n = multi_tau_auto_corr(4, num_bufs,
                                            rois, images)

    assert_array_almost_equal(g2[:, 0], g2_n[:, 0], decimal=3)
    assert_array_almost_equal(g2[:, 1], g2_n[:, 1], decimal=3)


def test_one_time_from_two_time():
    num_lev = 1
    num_buf = 10  # must be even
    x_dim = 10
    y_dim = 10
    stack = 10
    imgs = np.random.randint(1, 3, (stack, x_dim, y_dim))
    roi = np.zeros_like(imgs[0])
    # make sure that the ROIs can be any integers greater than 1.
    # They do not have to start at 1 and be continuous
    roi[0:x_dim//10, 0:y_dim//10] = 5
    roi[x_dim//10:x_dim//5, y_dim//10:y_dim//5] = 3

    g2, lag_steps, _state = two_time_corr(roi, imgs, stack,
                                          num_buf, num_lev)

    one_time = one_time_from_two_time(g2)
    assert_array_almost_equal(one_time[0, :], np.array([1.0, 0.9, 0.8, 0.7,
                                                        0.6, 0.5, 0.4, 0.3,
                                                        0.2, 0.1]))


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
