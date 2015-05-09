from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
                           assert_array_almost_equal, assert_almost_equal)

from skxray.cdi import (_dist, gauss, convolution,
                        cal_relative_error, find_support, pi_support)


def dist_temp(dims):
    """
    Another way to create array with pixel value equals euclidian distance
    from array center.
    This is used for test purpose only.
    This is Xiaojing's original code for computing the squared distance and is
    very useful as a test to ensure that new code conforms to the original
    code, as this has been used to publish results.
    """
    new_array = np.zeros(dims)

    if np.size(dims) == 2:
        x_sq = (np.arange(dims[0]) - dims[0]//2)**2
        y_sq = (np.arange(dims[1]) - dims[1]//2)**2
        for j in range(dims[1]):
            new_array[:, j] = np.sqrt(x_sq + y_sq[j])

    if np.size(dims) == 3:
        x_sq = (np.arange(dims[0]) - dims[0]//2)**2
        y_sq = (np.arange(dims[1]) - dims[1]//2)**2
        z_sq = (np.arange(dims[2]) - dims[2]//2)**2
        for j in range(dims[1]):
            for k in range(dims[2]):
                new_array[:, j, k] = np.sqrt(x_sq + y_sq[j] + z_sq[k])

    return new_array


def test_dist():
    shape2D = [150, 100]
    data = _dist(shape2D)
    data1 = dist_temp(shape2D)
    assert_array_equal(data.shape, shape2D)
    assert_array_equal(data, data1)

    shape3D = [100, 200, 300]
    data = _dist(shape3D)
    data1 = dist_temp(shape3D)
    assert_array_equal(data.shape, shape3D)
    assert_array_equal(data, data1)


def test_gauss():
    shape2D = (100, 100)
    shape3D = (100, 200, 50)
    shape_list = [shape2D, shape3D]
    std = 10

    for v in shape_list:
        d = gauss(v, std)
        assert_almost_equal(0, np.mean(d), decimal=3)


def test_convolution():
    shape_list = [(100, 50), (100, 100, 100)]
    std1 = 5
    std2 = 10
    for v in shape_list:
        g1 = gauss(v, std1)
        g2 = gauss(v, std2)
        f = convolution(g1, g2)
        assert_almost_equal(0, np.mean(f), decimal=3)


def test_relative_error():
    shape_v = [3, 3]
    a1 = np.zeros(shape_v)
    a2 = np.ones(shape_v)

    e1 = cal_relative_error(a2, a1)
    assert_equal(e1, 1)

    e2 = cal_relative_error(a2, a2)
    assert_equal(e2, 0)


def test_find_support():
    shape_v = [100, 100]
    cenv = shape_v[0]/2
    r = 20
    a = np.zeros(shape_v)
    a[cenv-r:cenv+r, cenv-r:cenv+r] = 1.0
    sw_sigma = 0.5
    sw_threshold = 0.01

    new_sup, s_index, s_out_index = find_support(a, sw_sigma, sw_threshold)
    # the area of new support becomes larger
    assert(np.sum(new_sup)-np.sum(a) > 0)


def test_pi_support():
    a1 = np.ones([2, 2])
    a1[0, 0] = 1
    index = np.where(a1 == 1)
    a2 = pi_support(a1, index)
    assert_equal(np.sum(a2), 0)
