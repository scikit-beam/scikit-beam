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
import os
import sys

import numpy as np
import numpy.testing as npt
import pytest
import six
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal, assert_equal

import skbeam.core.utils as core

try:
    from pyFAI.geometry import Geometry

    pf = True
except ImportError:
    pf = False

logger = logging.getLogger(__name__)


def test_bin_1D():
    # set up simple data
    x = np.linspace(0, 1, 100)
    y = np.arange(100)
    nx = 10
    # make call
    edges, val, count = core.bin_1D(x, y, nx)
    # check that values are as expected
    assert_array_almost_equal(edges, np.linspace(0, 1, nx + 1, endpoint=True))
    assert_array_almost_equal(val, np.sum(y.reshape(nx, -1), axis=1))
    assert_array_equal(count, np.ones(nx) * 10)


def test_bin_1D_2():
    """
    Test for appropriate default value handling
    """
    # set up simple data
    x = np.linspace(0, 1, 100)
    y = np.arange(100)
    nx = core._defaults["bins"]
    min_x = None
    max_x = None
    # make call
    edges, val, count = core.bin_1D(x=x, y=y, nx=nx, min_x=min_x, max_x=max_x)
    # check that values are as expected
    assert_array_almost_equal(edges, np.linspace(0, 1, nx + 1, endpoint=True))
    assert_array_almost_equal(val, np.sum(y.reshape(nx, -1), axis=1))
    assert_array_equal(count, np.ones(nx))


def test_bin_1D_limits():
    # set up simple data
    x = np.linspace(0, 1, 100)
    y = np.arange(100)
    nx = 10
    min_x, max_x = 0.25, 0.75
    # make call
    edges, val, count = core.bin_1D(x, y, nx, min_x, max_x)
    # check that values are as expected
    assert_array_almost_equal(edges, np.linspace(min_x, max_x, nx + 1, endpoint=True))
    assert_array_almost_equal(val, np.sum(y[25:75].reshape(nx, -1), axis=1))
    assert_array_equal(count, np.ones(nx) * 5)


def _bin_edges_helper(p_dict):
    bin_edges = core.bin_edges(**p_dict)
    assert_almost_equal(0, np.ptp(np.diff(bin_edges)))
    if "nbins" in p_dict:
        nbins = p_dict["nbins"]
        assert_equal(nbins + 1, len(bin_edges))
    if "step" in p_dict:
        step = p_dict["step"]
        assert_almost_equal(step, np.diff(bin_edges))
    if "range_max" in p_dict:
        range_max = p_dict["range_max"]
        assert np.all(bin_edges <= range_max)
    if "range_min" in p_dict:
        range_min = p_dict["range_min"]
        assert np.all(bin_edges >= range_min)
    if "range_max" in p_dict and "step" in p_dict:
        step = p_dict["step"]
        range_max = p_dict["range_max"]
        assert (range_max - bin_edges[-1]) < step


def _bin_edges_exceptions(param_dict):
    with pytest.raises(ValueError):
        core.bin_edges(**param_dict)


param_test_bin_edges = ["range_min", "range_max", "step", "nbins"]


@pytest.mark.parametrize("drop_key", param_test_bin_edges)
def test_bin_edges(drop_key):
    test_dict = {"range_min": 1.234, "range_max": 5.678, "nbins": 42, "step": np.pi / 10}
    test_dict.pop(drop_key)
    _bin_edges_helper(test_dict)


param_test_bin_edges_exceptions = [
    # no entries
    {},
    # 4 entries
    {"range_min": 1.234, "range_max": 5.678, "nbins": 42, "step": np.pi / 10},
    # 2 entries
    {"range_min": 1.234, "step": np.pi / 10},
    # 1 entry
    {
        "range_min": 1.234,
    },
    # max < min
    {"range_max": 1.234, "range_min": 5.678, "step": np.pi / 10},
    # step > max - min
    {"range_min": 1.234, "range_max": 5.678, "step": np.pi * 10},
    # nbins == 0
    {"range_min": 1.234, "range_max": 5.678, "nbins": 0},
]


@pytest.mark.parametrize("fail_dict", param_test_bin_edges_exceptions)
def test_bin_edges_exceptions(fail_dict):
    _bin_edges_exceptions(fail_dict)


@pytest.mark.skipif(os.name == "nt", reason="Test is not supported on Windows")
def test_grid3d():
    size = 10
    q_max = np.array([1.0, 1.0, 1.0])
    q_min = np.array([-1.0, -1.0, -1.0])
    dqn = np.array([size, size, size])
    param_dict = {
        "nx": dqn[0],
        "ny": dqn[1],
        "nz": dqn[2],
        "xmin": q_min[0],
        "ymin": q_min[1],
        "zmin": q_min[2],
        "xmax": q_max[0],
        "ymax": q_max[1],
        "zmax": q_max[2],
    }
    # slice tricks
    # this make a list of slices, the imaginary value in the
    # step is interpreted as meaning 'this many values'
    slc = [
        slice(_min + (_max - _min) / (s * 2), _max - (_max - _min) / (s * 2), 1j * s)
        for _min, _max, s in zip(q_min, q_max, dqn)
    ]
    # use the numpy slice magic to make X, Y, Z these are dense meshes with
    # points in the center of each bin
    X, Y, Z = np.mgrid[slc]

    # make and ravel the image data (which is all ones)
    II = np.ones_like(X).ravel()

    # make input data (Nx3
    data = np.array([np.ravel(X), np.ravel(Y), np.ravel(Z)]).T

    (mean, occupancy, std_err, bounds) = core.grid3d(data, II, **param_dict)

    # check the values are as expected
    npt.assert_array_equal(mean.ravel(), II)
    npt.assert_array_equal(occupancy, np.ones_like(occupancy))
    npt.assert_array_equal(std_err, 0)


@pytest.mark.skipif(os.name == "nt", reason="Test is not supported on Windows")
def test_process_grid_std_err():
    size = 10
    q_max = np.array([1.0, 1.0, 1.0])
    q_min = np.array([-1.0, -1.0, -1.0])
    dqn = np.array([size, size, size])
    param_dict = {
        "nx": dqn[0],
        "ny": dqn[1],
        "nz": dqn[2],
        "xmin": q_min[0],
        "ymin": q_min[1],
        "zmin": q_min[2],
        "xmax": q_max[0],
        "ymax": q_max[1],
        "zmax": q_max[2],
    }
    # slice tricks
    # this make a list of slices, the imaginary value in the
    # step is interpreted as meaning 'this many values'
    slc = [
        slice(_min + (_max - _min) / (s * 2), _max - (_max - _min) / (s * 2), 1j * s)
        for _min, _max, s in zip(q_min, q_max, dqn)
    ]
    # use the numpy slice magic to make X, Y, Z these are dense meshes with
    # points in the center of each bin
    X, Y, Z = np.mgrid[slc]

    # make and ravel the image data (which is all ones)
    I = np.hstack([j * np.ones_like(X).ravel() for j in range(1, 101)])  # noqa: E741

    # make input data (N*5x3)
    data = np.vstack([np.tile(_, 100) for _ in (np.ravel(X), np.ravel(Y), np.ravel(Z))]).T
    (mean, occupancy, std_err, bounds) = core.grid3d(data, I, **param_dict)

    # check the values are as expected
    npt.assert_array_equal(mean, np.ones_like(X) * np.mean(np.arange(1, 101)))
    npt.assert_array_equal(occupancy, np.ones_like(occupancy) * 100)
    # need to convert std -> ste (standard error)
    # according to wikipedia ste = std/sqrt(n)
    npt.assert_array_almost_equal(std_err, (np.ones_like(occupancy) * np.std(np.arange(1, 101)) / np.sqrt(100)))


def test_bin_edge2center():
    test_edges = np.arange(11)
    centers = core.bin_edges_to_centers(test_edges)
    assert_array_almost_equal(0.5, centers % 1)
    assert_equal(10, len(centers))


def test_small_verbosedict():
    expected_string = "You tried to access the key 'b' " "which does not exist.  " "The extant keys are: ['a']"
    dd = core.verbosedict()
    dd["a"] = 1
    assert_equal(dd["a"], 1)
    try:
        dd["b"]
    except KeyError as e:
        assert_equal(eval(six.text_type(e)), expected_string)
    else:
        # did not raise a KeyError
        assert False


def test_large_verbosedict():
    expected_sting = (
        "You tried to access the key 'a' "
        "which does not exist.  There are 100 "
        "extant keys, which is too many to show you"
    )

    dd = core.verbosedict()
    for j in range(100):
        dd[j] = j
    # test success
    for j in range(100):
        assert_equal(dd[j], j)
    # test failure
    try:
        dd["a"]
    except KeyError as e:
        assert_equal(eval(six.text_type(e)), expected_sting)
    else:
        # did not raise a KeyError
        assert False


def test_d_q_conversion():
    assert_equal(2 * np.pi, core.d_to_q(1))
    assert_equal(2 * np.pi, core.q_to_d(1))
    test_data = np.linspace(0.1, 5, 100)
    assert_array_almost_equal(test_data, core.d_to_q(core.q_to_d(test_data)), decimal=12)
    assert_array_almost_equal(test_data, core.q_to_d(core.d_to_q(test_data)), decimal=12)


def test_q_twotheta_conversion():
    wavelength = 1
    q = np.linspace(0, 4 * np.pi, 100)
    assert_array_almost_equal(q, core.twotheta_to_q(core.q_to_twotheta(q, wavelength), wavelength), decimal=12)
    two_theta = np.linspace(0, np.pi, 100)
    assert_array_almost_equal(
        two_theta, core.q_to_twotheta(core.twotheta_to_q(two_theta, wavelength), wavelength), decimal=12
    )


def test_radius_to_twotheta():
    dist_sample = 100
    radius = np.linspace(50, 100)

    two_theta = np.array(
        [
            0.46364761,
            0.47177751,
            0.47984053,
            0.48783644,
            0.49576508,
            0.5036263,
            0.51142,
            0.51914611,
            0.52680461,
            0.53439548,
            0.54191875,
            0.54937448,
            0.55676277,
            0.56408372,
            0.57133748,
            0.57852421,
            0.58564412,
            0.5926974,
            0.59968432,
            0.60660511,
            0.61346007,
            0.62024949,
            0.62697369,
            0.63363301,
            0.6402278,
            0.64675843,
            0.65322528,
            0.65962874,
            0.66596924,
            0.67224718,
            0.67846301,
            0.68461716,
            0.6907101,
            0.69674228,
            0.70271418,
            0.70862627,
            0.71447905,
            0.720273,
            0.72600863,
            0.73168643,
            0.73730693,
            0.74287063,
            0.74837805,
            0.75382971,
            0.75922613,
            0.76456784,
            0.76985537,
            0.77508925,
            0.78027,
            0.78539816,
        ]
    )

    assert_array_almost_equal(two_theta, core.radius_to_twotheta(dist_sample, radius), decimal=8)


def test_multi_tau_lags():
    levels = 3
    channels = 8

    delay_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28]
    dict_dly = {}
    dict_dly[1] = (0, 1, 2, 3, 4, 5, 6, 7)
    dict_dly[3] = (16, 20, 24, 28)
    dict_dly[2] = (8, 10, 12, 14)
    tot_channels, lag_steps, dict_lags = core.multi_tau_lags(levels, channels)

    assert_array_equal(16, tot_channels)
    assert_array_equal(delay_steps, lag_steps)
    assert_array_equal(dict_dly[1], dict_lags[1])
    assert_array_equal(dict_dly[3], dict_lags[3])


def test_wedge_integration():
    with pytest.raises(NotImplementedError):
        core.wedge_integration(
            src_data=None, center=None, theta_start=None, delta_theta=None, r_inner=None, delta_r=None
        )


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
        print("is_dark_lst: {0}".format(is_dark_lst))
        print("num_expected_images: {0}".format(num_expected_images))
        print("len(subtracted): {0}".format(len(subtracted)))
        six.reraise(AssertionError, ae, sys.exc_info()[2])
    # test that the image subtraction values are behaving as expected
    img_sum_lst = [img_dims * img_dims * val for val in range(num_images)]
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
        print("is_dark_lst: {0}".format(is_dark_lst))
        print("expected_return_val: {0}".format(expected_return_val))
        print("return_sum: {0}".format(return_sum))
        six.reraise(AssertionError, ae, sys.exc_info()[2])


def _fail_img_to_relative_xyi_helper(input_dict):
    with pytest.raises(ValueError):
        core.img_to_relative_xyi(**input_dict)


param_test_img_to_relative_fails = [
    # invalid values of x and y
    {"img": np.ones((100, 100)), "cx": 50, "cy": 50, "pixel_size_x": -1, "pixel_size_y": -1},
    # valid value of x, no value for y
    {"img": np.ones((100, 100)), "cx": 50, "cy": 50, "pixel_size_x": 1},
    # valid value of y, no value for x
    {"img": np.ones((100, 100)), "cx": 50, "cy": 50, "pixel_size_y": 1},
    # valid value of y, invalid value for x
    {"img": np.ones((100, 100)), "cx": 50, "cy": 50, "pixel_size_x": -1, "pixel_size_y": 1},
    # valid value of x, invalid value for y
    {"img": np.ones((100, 100)), "cx": 50, "cy": 50, "pixel_size_x": 1, "pixel_size_y": -1},
    # invalid value of x, no value for y
    {"img": np.ones((100, 100)), "cx": 50, "cy": 50, "pixel_size_x": -1},
    # invalid value of y, no value for x
    {"img": np.ones((100, 100)), "cx": 50, "cy": 50, "pixel_size_y": -1},
]


@pytest.mark.parametrize("fail_dict", param_test_img_to_relative_fails)
def test_img_to_relative_fails(fail_dict):
    _fail_img_to_relative_xyi_helper(fail_dict)


def test_img_to_relative_xyi(random_seed=None):
    from skbeam.core.utils import img_to_relative_xyi

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
        logger.debug("y {0}".format(y))
        logger.debug("sum(y) {0}".format(sum(y)))
        expected_total_y = sum(np.arange(ny, dtype=np.int64) - cy) * nx
        logger.debug("expected_total_y {0}".format(expected_total_y))
        logger.debug("x {0}".format(x))
        logger.debug("sum(x) {0}".format(sum(x)))
        expected_total_x = sum(np.arange(nx, dtype=np.int64) - cx) * ny
        logger.debug("expected_total_x {0}".format(expected_total_x))
        expected_total_intensity = nx * ny
        try:
            assert_almost_equal(sum(x), expected_total_x, decimal=0)
            assert_almost_equal(sum(y), expected_total_y, decimal=0)
            assert_equal(sum(i), expected_total_intensity)
        except AssertionError as ae:
            logger.error("img dims: ({0}, {1})".format(nx, ny))
            logger.error("img center: ({0}, {1})".format(cx, cy))
            logger.error("sum(returned_x): {0}".format(sum(x)))
            logger.error("expected_x: {0}".format(expected_total_x))
            logger.error("sum(returned_y): {0}".format(sum(y)))
            logger.error("expected_y: {0}".format(expected_total_y))
            logger.error("sum(returned_i): {0}".format(sum(i)))
            logger.error("expected_x: {0}".format(expected_total_intensity))
            six.reraise(AssertionError, ae, sys.exc_info()[2])


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
            print("{0} calls successful".format(num_calls))


def test_angle_grid():
    a = core.angle_grid((3, 3), (7, 7))
    assert_equal(a[3, -1], 0)
    assert_almost_equal(a[3, 0], np.pi)
    assert_almost_equal(a[4, 4], np.pi / 4)  # (1, 1) should be 45 degrees
    # The documented domain is [-pi, pi].
    correct_domain = np.all((a < np.pi + 0.1) & (a > -np.pi - 0.1))
    assert correct_domain


def test_radial_grid():
    a = core.radial_grid((3, 3), (7, 7))
    assert_equal(a[3, 3], 0)
    assert_equal(a[3, 4], 1)


def test_geometric_series():
    time_series = core.geometric_series(common_ratio=5, number_of_images=150)

    assert_array_equal(time_series, [1, 5, 25, 125])


def test_bin_grid():
    if not pf:
        pytest.skip("'Geometry' can not be imported from 'pyFAI.geometry'.")
    geo = Geometry(
        detector="Perkin",
        pixel1=0.0002,
        pixel2=0.0002,
        dist=0.23,
        poni1=0.209,
        poni2=0.207,
        # poni1=0, poni2=0,
        # rot1=.0128, rot2=-.015, rot3=-5.2e-8,
        wavelength=1.43e-11,
    )
    r_array = geo.rArray((2048, 2048))
    img = r_array.copy()
    x, y = core.bin_grid(img, r_array, (geo.pixel1, geo.pixel2))

    assert_array_almost_equal(y, x, decimal=2)
