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

import nsls2.core as core

from nsls2.testing.decorators import known_fail_if
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


def test_bin_image_to_1D_radius():
    shape = (256, 300)
    center = (120, 150)
    R = core.pixel_to_radius(shape, center)

    I = np.zeros_like(R, dtype='int')

    ring_width = 2

    ring_locs = [10, 50, 76]
    for r in ring_locs:
        I += ((R >= r) * (R < (r + ring_width))) * r

    A, B, C = core.bin_image_to_1D(I, center,
                    core.pixel_to_radius,
                    bin_min=0, bin_max=100,
                    bin_num=50)

    for j, (a, b, c) in enumerate(zip(A, B, C)):
        if j*2 in ring_locs:
            assert b == j * 2 * c
        else:
            assert b == 0


def test_bin_image_to_1D_phi():
    shape = (256, 300)
    center = (120, 150)
    phi = core.pixel_to_phi(shape, center)

    nphi_steps = 25

    I = np.zeros_like(phi, dtype='int')

    phi_steps = np.linspace(-np.pi, np.pi + np.spacing(np.pi),
                            nphi_steps + 1,
                            endpoint=True)
    for j, (bot, top) in enumerate(core.pairwise(phi_steps)):
        mask = (phi >= bot) * (phi < top)
        I[mask] = j + 1

    A, B, C = core.bin_image_to_1D(I, center,
                    core.pixel_to_phi,
                    bin_min=-np.pi, bin_max=np.pi,
                    bin_num=nphi_steps)

    for j, (a, b, c) in enumerate(zip(A, B, C)):
        assert b == c * (j + 1)


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

    two_theta = np.array([0.92729522, 0.94355502, 0.95968105,
                          0.97567288, 0.99153015, 1.00725259,
                          1.02284, 1.03829223, 1.05360922,
                          1.06879095, 1.0838375, 1.09874897,
                          1.11352554, 1.12816744, 1.14267496,
                          1.15704843, 1.17128823, 1.18539481,
                          1.19936863, 1.21321022, 1.22692013,
                          1.24049897, 1.25394738, 1.26726602,
                          1.2804556, 1.29351685, 1.30645055,
                          1.31925749, 1.33193847, 1.34449436,
                          1.35692602, 1.36923433, 1.3814202,
                          1.39348456, 1.40542836, 1.41725254,
                          1.4289581, 1.440546, 1.45201725,
                          1.46337287, 1.47461386, 1.48574126,
                          1.4967561, 1.50765941, 1.51845226,
                          1.52913569, 1.53971075, 1.5501785,
                          1.56054001, 1.57079633])

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
    from nsls2.core import img_to_relative_xyi
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


if __name__ == "__main__":
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
