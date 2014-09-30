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
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)

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


def test_multi_tau_lags():
    multi_tau_levels = 3
    multi_tau_channels = 8

    delay_steps = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32]

    tot_channels, lag_steps = core.multi_tau_lags(multi_tau_levels, multi_tau_channels)

    assert_almost_equal(16, tot_channels)
    assert_array_almost_equal(delay_steps, lag_steps)
