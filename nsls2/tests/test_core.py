from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)

from nose.tools import assert_equal, assert_true, raises

import nsls2.core as core


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
def _bin_edges_exceptions(p_dict):
    core.bin_edges(**p_dict)


def test_bin_edges():
    test_dicts = [{'range_min': 1.234,
                   'range_max': 5.678,
                   'nbins': 42,
                   'step': np.pi / 10}, ]
    for p_dict in test_dicts:
        for drop_key in ['range_min', 'range_max', 'step', 'nbins']:
            tmp_pdict = dict(p_dict)
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

    for p_dict in fail_dicts:
        yield _bin_edges_exceptions, p_dict
