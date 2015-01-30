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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
import logging
logger = logging.getLogger(__name__)
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)
import sys

from nose.tools import assert_equal, assert_true, raises

import skxray.core as core

from skxray.testing.decorators import known_fail_if
import numpy.testing as npt


def test_bin_1D():
    # set up simple data
    x = np.linspace(0, 1, 100)
    y = np.arange(100)
    nx = 10
    # make call
    edges, val, count = core.bin_1D(x, y, nx)
    # check that values are as expected
    assert_array_almost_equal(edges,
                              np.linspace(0, 1, nx + 1, endpoint=True))
    assert_array_almost_equal(val,
                              np.sum(y.reshape(nx, -1), axis=1))
    assert_array_equal(count,
                       np.ones(nx) * 10)


def test_bin_1D_2():
    """
    Test for appropriate default value handling
    """
    # set up simple data
    x = np.linspace(0, 1, 100)
    y = np.arange(100)
    nx = None
    min_x = None
    max_x = None
    # make call
    edges, val, count = core.bin_1D(x=x, y=y, nx=nx, min_x=min_x, max_x=max_x)
    # check that values are as expected
    nx = core._defaults["bins"]
    assert_array_almost_equal(edges,
                              np.linspace(0, 1, nx + 1, endpoint=True))
    assert_array_almost_equal(val,
                              np.sum(y.reshape(nx, -1), axis=1))
    assert_array_equal(count,
                       np.ones(nx))


def test_bin_1D_limits():
    # set up simple data
    x = np.linspace(0, 1, 100)
    y = np.arange(100)
    nx = 10
    min_x, max_x = .25, .75
    # make call
    edges, val, count = core.bin_1D(x, y, nx, min_x, max_x)
    # check that values are as expected
    assert_array_almost_equal(edges,
                              np.linspace(min_x, max_x, nx + 1, endpoint=True))
    assert_array_almost_equal(val,
                              np.sum(y[25:75].reshape(nx, -1), axis=1))
    assert_array_equal(count,
                       np.ones(nx) * 5)


def _bin_edges_helper(p_dict):
    bin_edges = core.bin_edges(**p_dict)
    assert_almost_equal(0, np.ptp(np.diff(bin_edges)))
    if 'nbins' in p_dict:
        nbins = p_dict['nbins']
        assert_equal(nbins + 1, len(bin_edges))
    if 'step' in p_dict:
        step = p_dict['step']
        assert_almost_equal(step, np.diff(bin_edges))
    if 'range_max' in p_dict:
        range_max = p_dict['range_max']
        assert_true(np.all(bin_edges <= range_max))
    if 'range_min' in p_dict:
        range_min = p_dict['range_min']
        assert_true(np.all(bin_edges >= range_min))
    if 'range_max' in p_dict and 'step' in p_dict:
        step = p_dict['step']
        range_max = p_dict['range_max']
        assert_true((range_max - bin_edges[-1]) < step)


@raises(ValueError)
def _bin_edges_exceptions(param_dict):
    core.bin_edges(**param_dict)


def test_bin_edges():
    test_dicts = [{'range_min': 1.234,
                   'range_max': 5.678,
                   'nbins': 42,
                   'step': np.pi / 10}, ]
    for param_dict in test_dicts:
        for drop_key in ['range_min', 'range_max', 'step', 'nbins']:
            tmp_pdict = dict(param_dict)
            tmp_pdict.pop(drop_key)
            yield _bin_edges_helper, tmp_pdict

    fail_dicts = [{},  # no entries

                  {'range_min': 1.234,
                   'range_max': 5.678,
                   'nbins': 42,
                   'step': np.pi / 10},  # 4 entries

                   {'range_min': 1.234,
                    'step': np.pi / 10},  # 2 entries

                    {'range_min': 1.234, },  # 1 entry

                   {'range_max': 1.234,
                   'range_min': 5.678,
                   'step': np.pi / 10},   # max < min

                   {'range_min': 1.234,
                   'range_max': 5.678,
                   'step': np.pi * 10},  # step > max - min

                   {'range_min': 1.234,
                   'range_max': 5.678,
                   'nbins': 0},  # nbins == 0
                ]

    for param_dict in fail_dicts:
        yield _bin_edges_exceptions, param_dict




@known_fail_if(six.PY3)
def test_grid3d():
    size = 10
    q_max = np.array([1.0, 1.0, 1.0])
    q_min = np.array([-1.0, -1.0, -1.0])
    dqn = np.array([size, size, size])
    param_dict = {'nx': dqn[0],
                  'ny': dqn[1],
                  'nz': dqn[2],
                  'xmin': q_min[0],
                  'ymin': q_min[1],
                  'zmin': q_min[2],
                  'xmax': q_max[0],
                  'ymax': q_max[1],
                  'zmax': q_max[2]}
    # slice tricks
    # this make a list of slices, the imaginary value in the
    # step is interpreted as meaning 'this many values'
    slc = [slice(_min + (_max - _min)/(s * 2),
                 _max - (_max - _min)/(s * 2),
                 1j * s)
           for _min, _max, s in zip(q_min, q_max, dqn)]
    # use the numpy slice magic to make X, Y, Z these are dense meshes with
    # points in the center of each bin
    X, Y, Z = np.mgrid[slc]

    # make and ravel the image data (which is all ones)
    I = np.ones_like(X).ravel()

    # make input data (Nx3
    data = np.array([np.ravel(X),
                     np.ravel(Y),
                     np.ravel(Z)]).T

    (mean, occupancy,
     std_err, oob, bounds) = core.grid3d(data, I, **param_dict)

    # check the values are as expected
    npt.assert_array_equal(mean.ravel(), I)
    npt.assert_equal(oob, 0)
    npt.assert_array_equal(occupancy, np.ones_like(occupancy))
    npt.assert_array_equal(std_err, 0)


@known_fail_if(six.PY3)
def test_process_grid_std_err():
    size = 10
    q_max = np.array([1.0, 1.0, 1.0])
    q_min = np.array([-1.0, -1.0, -1.0])
    dqn = np.array([size, size, size])
    param_dict = {'nx': dqn[0],
                  'ny': dqn[1],
                  'nz': dqn[2],
                  'xmin': q_min[0],
                  'ymin': q_min[1],
                  'zmin': q_min[2],
                  'xmax': q_max[0],
                  'ymax': q_max[1],
                  'zmax': q_max[2]}
    # slice tricks
    # this make a list of slices, the imaginary value in the
    # step is interpreted as meaning 'this many values'
    slc = [slice(_min + (_max - _min)/(s * 2),
                 _max - (_max - _min)/(s * 2),
                 1j * s)
           for _min, _max, s in zip(q_min, q_max, dqn)]
    # use the numpy slice magic to make X, Y, Z these are dense meshes with
    # points in the center of each bin
    X, Y, Z = np.mgrid[slc]

    # make and ravel the image data (which is all ones)
    I = np.hstack([j * np.ones_like(X).ravel() for j in range(1, 6)])

    # make input data (N*5x3)
    data = np.vstack([np.tile(_, 5)
                      for _ in (np.ravel(X), np.ravel(Y), np.ravel(Z))]).T
    (mean, occupancy,
     std_err, oob, bounds) = core.grid3d(data, I, **param_dict)

    # check the values are as expected
    npt.assert_array_equal(mean,
                           np.ones_like(X) * np.mean(np.arange(1, 6)))
    npt.assert_equal(oob, 0)
    npt.assert_array_equal(occupancy, np.ones_like(occupancy)*5)
    # need to convert std -> ste (standard error)
    # according to wikipedia ste = std/sqrt(n), but experimentally, this is
    # implemented as ste = std / srt(n - 1)
    npt.assert_array_equal(std_err,
                           (np.ones_like(occupancy) *
                            np.std(np.arange(1, 6))/np.sqrt(5 - 1)))


def test_bin_edge2center():
    test_edges = np.arange(11)
    centers = core.bin_edges_to_centers(test_edges)
    assert_array_almost_equal(.5, centers % 1)
    assert_equal(10, len(centers))


def test_small_verbosedict():
    if six.PY2:
        expected_string = ("You tried to access the key 'b' "
                       "which does not exist.  "
                       "The extant keys are: [u'a']")
    elif six.PY3:
        expected_string = ("You tried to access the key 'b' "
                       "which does not exist.  "
                       "The extant keys are: ['a']")
    else:
        # should never happen....
        assert(False)
    dd = core.verbosedict()
    dd['a'] = 1
    assert_equal(dd['a'], 1)
    try:
        dd['b']
    except KeyError as e:
        assert_equal(eval(six.text_type(e)), expected_string)
    else:
        # did not raise a KeyError
        assert(False)


def test_large_verbosedict():
    expected_sting = ("You tried to access the key 'a' "
                      "which does not exist.  There are 100 "
                      "extant keys, which is too many to show you")

    dd = core.verbosedict()
    for j in range(100):
        dd[j] = j
    # test success
    for j in range(100):
        assert_equal(dd[j], j)
    # test failure
    try:
        dd['a']
    except KeyError as e:
        assert_equal(eval(six.text_type(e)), expected_sting)
    else:
        # did not raise a KeyError
        assert(False)


def test_d_q_conversion():
    assert_equal(2 * np.pi, core.d_to_q(1))
    assert_equal(2 * np.pi, core.q_to_d(1))
    test_data = np.linspace(.1, 5, 100)
    assert_array_almost_equal(test_data, core.d_to_q(core.q_to_d(test_data)),
                              decimal=12)
    assert_array_almost_equal(test_data, core.q_to_d(core.d_to_q(test_data)),
                              decimal=12)


def test_q_twotheta_conversion():
    wavelength = 1
    q = np.linspace(0, 4 * np.pi, 100)
    assert_array_almost_equal(q,
                              core.twotheta_to_q(
                                  core.q_to_twotheta(q, wavelength),
                                  wavelength),
                              decimal=12)
    two_theta = np.linspace(0, np.pi, 100)
    assert_array_almost_equal(two_theta,
                              core.q_to_twotheta(
                                  core.twotheta_to_q(two_theta,
                                                     wavelength),
                                  wavelength),
                              decimal=12)


def test_radius_to_twotheta():
    dist_sample = 100
    radius = np.linspace(50, 100)

    two_theta = np.array([0.46364761, 0.47177751, 0.47984053, 0.48783644, 0.49576508,
                          0.5036263, 0.51142, 0.51914611, 0.52680461, 0.53439548,
                          0.54191875, 0.54937448, 0.55676277, 0.56408372, 0.57133748,
                          0.57852421, 0.58564412, 0.5926974, 0.59968432, 0.60660511,
                          0.61346007, 0.62024949, 0.62697369, 0.63363301, 0.6402278,
                          0.64675843, 0.65322528, 0.65962874, 0.66596924, 0.67224718,
                          0.67846301, 0.68461716, 0.6907101, 0.69674228, 0.70271418,
                          0.70862627, 0.71447905, 0.720273, 0.72600863, 0.73168643,
                          0.73730693, 0.74287063, 0.74837805, 0.75382971, 0.75922613,
                          0.76456784, 0.76985537, 0.77508925, 0.78027, 0.78539816])

    assert_array_almost_equal(two_theta,
                              core.radius_to_twotheta(dist_sample,
                                                      radius), decimal=8)


def test_multi_tau_lags():
    multi_tau_levels = 3
    multi_tau_channels = 8

    delay_steps = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32]

    tot_channels, lag_steps = core.multi_tau_lags(multi_tau_levels,
                                                  multi_tau_channels)

    assert_array_equal(16, tot_channels)
    assert_array_equal(delay_steps, lag_steps)


@raises(NotImplementedError)
def test_wedge_integration():
    core.wedge_integration(src_data=None, center=None, theta_start=None,
                           delta_theta=None, r_inner=None, delta_r=None)


def test_subtract_reference_images():
    num_images = 10
    img_dims = 200
    ones = np.ones((img_dims, img_dims))
    img_lst = [ones * _ for _ in range(num_images)]
    img_arr = np.asarray(img_lst)
    is_dark_lst = [True]
    is_dark = False
    was_dark = True
    while len(is_dark_lst) < num_images:
        if was_dark:
            is_dark = False
        else:
            is_dark = np.random.rand() > 0.5
        was_dark = is_dark
        is_dark_lst.append(is_dark)

    is_dark_arr = np.asarray(is_dark_lst)
    # make sure that a list of 2d images can be passed in
    core.subtract_reference_images(imgs=img_lst, is_reference=is_dark_arr)
    # make sure that the reference arr can actually be a list
    core.subtract_reference_images(imgs=img_arr, is_reference=is_dark_lst)
    # make sure that both input arrays can actually be lists
    core.subtract_reference_images(imgs=img_arr, is_reference=is_dark_lst)

    # test that the number of returned images is equal to the expected number
    # of returned images
    num_expected_images = is_dark_lst.count(False)
    # subtract an additional value if the last image is a reference image
    # num_expected_images -= is_dark_lst[len(is_dark_lst)-1]
    subtracted = core.subtract_reference_images(img_lst, is_dark_lst)
    try:
        assert_equal(num_expected_images, len(subtracted))
    except AssertionError as ae:
        print('is_dark_lst: {0}'.format(is_dark_lst))
        print('num_expected_images: {0}'.format(num_expected_images))
        print('len(subtracted): {0}'.format(len(subtracted)))
        six.reraise(AssertionError, ae, sys.exc_info()[2])
    # test that the image subtraction values are behaving as expected
    img_sum_lst = [img_dims * img_dims * val for val in range(num_images)]
    total_val = sum(img_sum_lst)
    expected_return_val = 0
    dark_val = 0
    for idx, (is_dark, img_val) in enumerate(zip(is_dark_lst, img_sum_lst)):
        if is_dark:
            dark_val = img_val
        else:
            expected_return_val = expected_return_val - dark_val + img_val
    # test that the image subtraction was actually processed correctly
    return_sum = sum(subtracted)
    try:
        while True:
            return_sum = sum(return_sum)
    except TypeError:
        # thrown when return_sum is a single number
        pass

    try:
        assert_equal(expected_return_val, return_sum)
    except AssertionError as ae:
        print('is_dark_lst: {0}'.format(is_dark_lst))
        print('expected_return_val: {0}'.format(expected_return_val))
        print('return_sum: {0}'.format(return_sum))
        six.reraise(AssertionError, ae, sys.exc_info()[2])


@raises(ValueError)
def _fail_img_to_relative_xyi_helper(input_dict):
    core.img_to_relative_xyi(**input_dict)

def test_img_to_relative_fails():
    fail_dicts = [
        # invalid values of x and y
        {'img': np.ones((100, 100)),'cx': 50, 'cy': 50, 'pixel_size_x': -1, 'pixel_size_y': -1},
        # valid value of x, no value for y
        {'img': np.ones((100, 100)),'cx': 50, 'cy': 50, 'pixel_size_x': 1},
        # valid value of y, no value for x
        {'img': np.ones((100, 100)),'cx': 50, 'cy': 50, 'pixel_size_y': 1},
        # valid value of y, invalid value for x
        {'img': np.ones((100, 100)),'cx': 50, 'cy': 50, 'pixel_size_x': -1, 'pixel_size_y': 1},
        # valid value of x, invalid value for y
        {'img': np.ones((100, 100)),'cx': 50, 'cy': 50, 'pixel_size_x': 1, 'pixel_size_y': -1},
        # invalid value of x, no value for y
        {'img': np.ones((100, 100)),'cx': 50, 'cy': 50, 'pixel_size_x': -1,},
        # invalid value of y, no value for x
        {'img': np.ones((100, 100)),'cx': 50, 'cy': 50, 'pixel_size_y': -1,},
    ]
    for failer in fail_dicts:
        yield _fail_img_to_relative_xyi_helper, failer


def test_img_to_relative_xyi(random_seed=None):
    from skxray.core import img_to_relative_xyi
    # make the RNG deterministic
    if random_seed is not None:
        np.random.seed(42)
    # set the maximum image dims
    maxx = 2000
    maxy = 2000
    # create a randomly sized image
    nx = int(np.random.rand() * maxx)
    ny = int(np.random.rand() * maxy)
    # create a randomly located center
    cx = np.random.rand() * nx
    cy = np.random.rand() * ny
    # generate the image
    img = np.ones((nx, ny))
    # generate options for the x center to test edge conditions
    cx_lst = [0, cx, nx]
    # generate options for the y center to test edge conditions
    cy_lst = [0, cy, ny]
    for cx, cy in zip(cx_lst, cy_lst):
        # call the function
        x, y, i = img_to_relative_xyi(img=img, cx=cx, cy=cy)
        logger.debug('y {0}'.format(y))
        logger.debug('sum(y) {0}'.format(sum(y)))
        expected_total_y = sum(np.arange(ny, dtype=np.int64) - cy) * nx
        logger.debug('expected_total_y {0}'.format(expected_total_y))
        logger.debug('x {0}'.format(x))
        logger.debug('sum(x) {0}'.format(sum(x)))
        expected_total_x = sum(np.arange(nx, dtype=np.int64) - cx) * ny
        logger.debug('expected_total_x {0}'.format(expected_total_x))
        expected_total_intensity = nx * ny
        try:
            assert_almost_equal(sum(x), expected_total_x, decimal=0)
            assert_almost_equal(sum(y), expected_total_y, decimal=0)
            assert_equal(sum(i), expected_total_intensity)
        except AssertionError as ae:
            logger.error('img dims: ({0}, {1})'.format(nx, ny))
            logger.error('img center: ({0}, {1})'.format(cx, cy))
            logger.error('sum(returned_x): {0}'.format(sum(x)))
            logger.error('expected_x: {0}'.format(expected_total_x))
            logger.error('sum(returned_y): {0}'.format(sum(y)))
            logger.error('expected_y: {0}'.format(expected_total_y))
            logger.error('sum(returned_i): {0}'.format(sum(i)))
            logger.error('expected_x: {0}'.format(expected_total_intensity))
            six.reraise(AssertionError, ae, sys.exc_info()[2])


def test_roi_rectangles():
    detector_size = (15, 10)
    num_rois = 3
    roi_data = np.array(([2, 2, 3, 3], [6, 7, 3, 2], [11, 8, 5, 2]),
                        dtype=np.int64)

    xy_inds, num_pixels = core.roi_rectangles(num_rois, roi_data, detector_size)

    xy_inds_m =([0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 2, 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 2, 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 2, 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
                [0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
                [0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
                [0, 0, 0, 0, 0, 0, 0, 0, 3, 3])
    num_pixels_m = [9, 6, 8]

    assert_array_equal(num_pixels, num_pixels_m)
    assert_array_equal(xy_inds, np.ravel(xy_inds_m))


def run_image_to_relative_xyi_repeatedly():
    level = logging.ERROR
    ch = logging.StreamHandler()
    ch.setLevel(level)
    logger.addHandler(ch)
    logger.setLevel(level)

    num_calls = 0
    while True:
        test_img_to_relative_xyi()
        num_calls += 1
        if num_calls % 10 == 0:
            print('{0} calls successful'.format(num_calls))


def test_ring_val():
    calibrated_center = (0.5, 0.5)
    img_dim = (15, 12)
    first_q = 2.5
    delta_q = 2.5
    num_qs = 20  # number of Q rings

    (q_inds, q_ring_val, num_pixels, pixel_list) = core.roi_rings(img_dim,
                                                                  calibrated_center,
                                                                  num_qs,first_q,
                                                                  delta_q)

    q_inds_m = np.array([[0, 0, 1, 2, 5, 8, 12, 17, 0, 0, 0, 0],
                        [0, 0, 1, 2, 5, 8, 12, 17, 0, 0, 0, 0],
                        [1, 1, 1, 3, 5, 9, 13, 17, 0, 0, 0, 0],
                        [2, 2, 3,  5, 7, 10, 14, 19, 0, 0, 0, 0],
                        [5, 5, 5,  7, 9, 13, 17, 0, 0, 0, 0, 0],
                        [8, 8, 9, 10, 13, 16, 20, 0, 0, 0, 0, 0],
                        [12, 12, 13, 14, 17, 20, 0, 0, 0, 0, 0, 0],
                        [17, 17, 17, 19, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    q_ring_val_m = np.array([[2.5, 5.],
                             [5., 7.5],
                             [7.5, 10.],
                             [10., 12.5],
                             [12.5, 15.],
                             [15., 17.5],
                             [17.5, 20.],
                             [20., 22.5],
                             [22.5, 25.],
                             [25., 27.5],
                             [27.5, 30.],
                             [30., 32.5],
                             [32.5, 35.],
                             [35., 37.5],
                             [37.5, 40.],
                             [40., 42.5],
                             [42.5, 45.],
                             [45., 47.5],
                             [47.5, 50.],
                             [50., 52.5]])

    num_pixels_m = np.array(([5, 4, 2, 0, 7, 0, 2, 4, 3, 2, 0, 4, 4, 2,
                              0, 1, 8, 0, 2, 2]))

    pixel_list_m = np.array([30,  45, 60, 75, 90, 105, 31, 46, 61,
                             76, 91, 106, 2, 17,  32, 47, 62, 77,
                             92, 107, 3, 18, 33, 48, 63, 78, 93, 108,
                             4, 19, 34, 49, 64, 79, 94,  5, 20, 35, 50,
                             65,  80, 95, 6, 21, 36, 51, 66, 81, 7, 22,
                             37, 52])

    assert_array_almost_equal(q_ring_val_m, q_ring_val)
    assert_array_equal(num_pixels, num_pixels_m)
    assert_array_equal(q_inds, np.ravel(q_inds_m))
    assert_array_equal(pixel_list, pixel_list_m)

    # using a step for the Q rings
    (qstep_inds, qstep_ring_val, numstep_pixels,
     pixelstep_list) = core.roi_rings_stepl(img_dim, calibrated_center, num_qs,
               first_q, delta_q, 0.5)

    qstep_inds_m = np.array([[0, 0, 1, 2, 4, 7, 10, 14, 19, 0, 0, 0],
                            [0, 0, 1, 2, 4, 7, 10, 14, 19, 0, 0, 0],
                            [1, 1, 1, 3, 5, 7, 11, 15, 19, 0, 0, 0],
                            [2, 2, 3, 4, 6, 9, 12, 16, 0, 0, 0, 0],
                            [4, 4, 5, 6, 8, 11, 14, 18, 0, 0, 0, 0],
                            [7, 7, 7, 9, 11, 13, 17, 0, 0, 0, 0, 0],
                            [10, 10, 11, 12, 14, 17, 20, 0, 0, 0, 0, 0],
                            [14, 14, 15, 16, 18, 0, 0, 0, 0, 0, 0, 0],
                            [19, 19, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    numstep_pixels_m = np.array([5, 4, 2, 5, 2, 2, 6, 1, 2, 4, 4, 2, 1,
                                 6, 2, 2, 2, 2, 6, 1])

    qstep_ring_val_m = np.array([[2.5, 5.],
                                [5.5, 8.],
                                [8.5, 11.],
                                [11.5, 14.],
                                [14.5, 17.],
                                [17.5, 20.],
                                [20.5, 23.],
                                [23.5, 26.],
                                [26.5, 29.],
                                [29.5, 32.],
                                [32.5, 35.],
                                [35.5, 38.],
                                [38.5, 41.],
                                [41.5, 44.],
                                [44.5, 47.],
                                [47.5, 50.],
                                [50.5, 53.],
                                [53.5, 56.],
                                [56.5, 59.],
                                [59.5, 62.]])

    pixelstep_list_m = np.array([30, 45, 60, 75, 90, 105, 120, 31,
                                 46, 61, 76, 91, 106, 121,  2, 17,
                                 32, 47, 62, 77, 92, 107, 122, 3,
                                 18, 33, 48, 63, 78, 93, 108, 4,
                                 19, 34, 49, 64,  79, 94, 109, 5,
                                 20, 35, 50, 65, 80, 95, 6, 21, 36,
                                 51, 66, 81, 96,  7, 22, 37, 52, 67,
                                 8, 23, 38])

    assert_almost_equal(qstep_ring_val, qstep_ring_val_m)
    assert_array_equal(numstep_pixels, numstep_pixels_m)
    assert_array_equal(qstep_inds, np.ravel(qstep_inds_m))

    assert_array_equal(pixelstep_list, pixelstep_list_m)

    num_qs = 8
    (qd_inds, qd_ring_val, numd_pixels, pixeld_list) = core.roi_rings_step(img_dim,
                                                                           calibrated_center,
                                                                           num_qs, first_q,
                                                                           delta_q,
                                                                           0.4, 0.2, 0.5, 0.4,
                                                                           0.0, 0.6, 0.2)

    qd_ring_val_m = ([[2.5, 5.],
                     [5.4, 7.9],
                     [8.1, 10.6],
                     [11.1, 13.6],
                     [14., 16.5],
                     [16.5, 19.],
                     [19.6, 22.1],
                     [22.3, 24.8]])

    numd_pixels_m = np.array([5, 4, 2, 5, 2, 2, 4, 3])

    pixeld_list_m = np.array([30, 45, 60, 75, 31, 46, 61, 76, 2, 17, 32, 47,
                              62, 77,  3, 18, 33, 48, 63, 4, 19, 34, 49, 64,
                              5, 20, 35])

    qd_inds_m = np.array([[0, 0, 1, 2, 4, 7, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 2, 4, 7, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 3, 5, 8, 0, 0, 0, 0, 0, 0],
                         [2, 2, 3, 4, 6, 0, 0, 0, 0, 0, 0, 0],
                         [4, 4, 5, 6, 8, 0, 0, 0, 0, 0, 0, 0],
                         [7, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    assert_array_equal(qd_inds, np.ravel(qd_inds_m))
    assert_array_equal(qd_ring_val, qd_ring_val_m)
    assert_array_equal(numd_pixels, numd_pixels_m)
    assert_array_equal(pixeld_list, pixeld_list_m)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
