from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
                           assert_array_almost_equal, assert_almost_equal)

from skxray.cdi import (_dist, gauss, convolution,
                        cal_relative_error, find_support, pi_support,
                        _fft_helper, pi_modulus, cal_diff_error, CDI)


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


def test_fft_helper():
    x = np.linspace(-2, 2, 100, endpoint=True)
    g = np.exp(x ** 2 / 2)

    g_fft = _fft_helper(g)
    g_ifft = _fft_helper(g_fft, option='ifft')
    assert_array_almost_equal(np.abs(g_ifft), g, decimal=10)


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


def cal_diff():
    """
    Fft transform of a squared area.

    Returns
    -------
    a : array
        squared sample
    diff_v : array
        fft transform of sample area
    """
    shapev = [100, 100]
    r = 20
    a = np.zeros(shapev)
    a[shapev[0]//2-r:shapev[0]//2+r, shapev[1]//2-r:shapev[1]//2+r] = 1
    diff_v = np.abs(_fft_helper(a)) / np.sqrt(np.size(a))
    return a, diff_v


def run_pi_modulus(data_type):
    a, diff_v = cal_diff()
    a_new = pi_modulus(a, diff_v, data_type)
    assert_array_almost_equal(np.abs(a_new), a)


def test_pi_modulus():
    type_list = ['Real', 'Complex']
    for d in type_list:
        yield run_pi_modulus, d


def test_cal_diff_error():
    a, diff_v = cal_diff()
    result = cal_diff_error(a, diff_v)
    assert_equal(np.sum(result), 0)


def test_recon():
    a, diff_v = cal_diff()
    total_n = 10
    cdi_param = {'beta': 1.15,
                 'start_ave': 0.8,
                 'pi_modulus_flag': 'Complex',
                 'init_obj_flag': True,
                 'init_sup_flag': True,
                 'support_radius': 20,
                 'support_shape': 'Box',
                 'shrink_wrap_flag': False,
                 'sw_sigma': 0.5,
                 'sw_threshold': 0.1,
                 'sw_start': 0.2,
                 'sw_end': 0.8,
                 'sw_step': 10}
    cdi = CDI(diff_v, **cdi_param)
    outv = cdi.recon(n_iterations=total_n)
    outv = np.abs(outv)
    # compare the area of supports
    assert_almost_equal(np.shape(outv[outv>0.9]), np.shape(a[a>0.9]))
