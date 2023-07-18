from __future__ import division

import random
from time import time

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal

from skbeam.core.accumulators.histogram import Histogram


def _1d_histogram_tester(binlowhighs, x, weights=1):
    h = Histogram(binlowhighs)
    h.fill(x, weights=weights)
    if np.isscalar(weights):
        ynp = np.histogram(x, h.edges[0])[0]
    else:
        ynp = np.histogram(x, h.edges[0], weights=weights)[0]
    assert_array_almost_equal(ynp, h.values)
    h.reset()
    h._always_use_fillnd = True
    h.fill(x, weights=weights)
    assert_array_almost_equal(ynp, h.values)


def _gen_test_1d_histogram():
    binlowhigh = [10, 0, 10.01]
    xf = np.random.random(1000000) * 40
    xi = xf.astype(int)
    xl = xf.tolist()
    wf = np.linspace(1, 10, len(xf))
    wi = wf.copy()
    wl = wf.tolist()
    onexf = random.random() * binlowhigh[2]
    onexi = int(onexf)
    vals = [
        (binlowhigh, xf, wf),
        (binlowhigh, xf, 1),
        (binlowhigh, xi, wi),
        (binlowhigh, xi, 1),
        (binlowhigh, xf, wi),
        (binlowhigh, xi, wf),
        (binlowhigh, xl, wl),
        (binlowhigh, xi, wl),
        (binlowhigh, xl, wi),
        (binlowhigh, onexf, 1),
        (binlowhigh, onexi, 1),
    ]
    return vals


par_test_1d_histogram = _gen_test_1d_histogram()


@pytest.mark.parametrize("binlowhigh, x, w", par_test_1d_histogram)
def test_1d_histogram(binlowhigh, x, w):
    _1d_histogram_tester(binlowhigh, x, w)


def _2d_histogram_tester(binlowhighs, x, y, weights=1):
    h = Histogram(*binlowhighs)
    h.fill(x, y, weights=weights)
    if np.isscalar(weights):
        if np.isscalar(x):
            assert np.isscalar(y), "If x is a scalar, y must be also"
            ynp = np.histogram2d([x], [y], bins=h.edges)[0]
        else:
            ynp = np.histogram2d(x, y, bins=h.edges)[0]
    else:
        ynp = np.histogram2d(x, y, bins=h.edges, weights=weights)[0]
    assert_array_almost_equal(ynp, h.values)
    h.reset()
    h._always_use_fillnd = True
    h.fill(x, y, weights=weights)
    assert_array_almost_equal(ynp, h.values)


def _gen_test_2d_histogram():
    ten = [10, 0, 10.01]
    nine = [9, 0, 9.01]
    onexf = random.random() * ten[2]
    onexi = int(onexf)
    oneyf = random.random() * ten[2]
    oneyi = int(oneyf)
    xf = np.random.random(1000000) * 40
    yf = np.random.random(1000000) * 40
    xi = xf.astype(int)
    yi = yf.astype(int)
    xl = xf.tolist()
    yl = yf.tolist()
    wf = np.linspace(1, 10, len(xf))
    wi = wf.copy()
    wl = wf.tolist()
    vals = [
        ((ten, ten), xf, yf, wf),
        ((ten, nine), xf, yf, 1),
        ((ten, ten), xi, yi, wi),
        ((ten, ten), xi, yi, 1),
        ((ten, nine), xf, yf, wi),
        ((ten, nine), xi, yi, wf),
        ((ten, nine), xl, yl, wl),
        ((ten, nine), xi, yi, wl),
        ((ten, nine), xf, yf, wl),
        ((ten, nine), xl, yl, wi),
        ((ten, nine), xl, yl, wf),
        ((ten, nine), onexf, oneyf, 1),
        ((ten, nine), onexi, oneyi, 1),
    ]
    return vals


par_test_2d_histogram = _gen_test_2d_histogram()


@pytest.mark.parametrize("binlowhigh, x, y, w", par_test_2d_histogram)
def test_2d_histogram(binlowhigh, x, y, w):
    _2d_histogram_tester(binlowhigh, x, y, w)


def test_simple_fail():
    # This test exposes the half-open vs full-open histogram code difference
    # between np.histogram and skbeam's histogram.
    with pytest.raises(AssertionError):
        h = Histogram((5, 0, 3))
        a = np.arange(1, 6)
        b = np.asarray([1, 1, 2, 3, 2])
        h.fill(a, weights=b)
        np_res = np.histogram(a, h.edges[0], weights=b)[0]
        assert_array_equal(h.values, np_res)


def test_simple_pass():
    # This test exposes the half-open vs full-open histogram code difference
    # between np.histogram and skbeam's histogram.
    h = Histogram((5, 0, 3.1))
    a = np.arange(1, 6)
    b = np.asarray([1, 1, 2, 3, 2])
    h.fill(a, weights=b)
    np_res = np.histogram(a, h.edges[0], weights=b)[0]
    assert_array_equal(h.values, np_res)


if __name__ == "__main__":
    import itertools

    x = [1000, 0, 10.01]
    y = [1000, 0, 9.01]
    xf = np.random.random(1000000) * 10 * 4
    yf = np.random.random(1000000) * 9 * 15
    xi = xf.astype(int)
    yi = yf.astype(int)
    wf = np.linspace(1, 10, len(xf))
    wi = wf.copy()
    times = []
    print("Testing 2D histogram timings")
    for xvals, yvals, weights in itertools.product([xf, xi], [yf, yi], [wf, wi]):
        t0 = time()
        h = Histogram(x, y)
        h.fill(xvals, yvals, weights=weights)
        skbeam_time = time() - t0

        edges = h.edges
        t0 = time()
        ynp = np.histogram2d(xvals, yvals, bins=edges, weights=weights)[0]
        numpy_time = time() - t0
        times.append(numpy_time / skbeam_time)
        assert_almost_equal(np.sum(h.values), np.sum(ynp))
    print("skbeam is %s times faster than numpy, on average" % np.average(times))
    # test_1d_histogram()
    # test_2d_histogram()

# TODO do a better job sampling the variable space
