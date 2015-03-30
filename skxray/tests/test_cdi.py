from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)

from skxray.cdi import _dist, gauss, convolution


def dist_temp(dims):
    """
    Another way to create array with pixel value equals
    euclidian distance from array center.
    This is used for test purpose only.
    """
    vec = [np.abs(np.arange(d) - (d-1.)/2.) for d in dims]
    grid = np.sqrt(np.sum([g*g for g in np.meshgrid(*vec, indexing='ij')], axis=0))
    return grid


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
