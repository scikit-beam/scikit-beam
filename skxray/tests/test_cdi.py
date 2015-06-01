from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)

from skxray.cdi import _dist, gauss, convolution


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
