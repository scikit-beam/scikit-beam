from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal, assert_equal

from skbeam.core.cdi import (
    _dist,
    cal_diff_error,
    cdi_recon,
    find_support,
    gauss,
    generate_box_support,
    generate_disk_support,
    generate_random_phase_field,
    pi_modulus,
)


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
        x_sq = (np.arange(dims[0]) - dims[0] // 2) ** 2
        y_sq = (np.arange(dims[1]) - dims[1] // 2) ** 2
        for j in range(dims[1]):
            new_array[:, j] = np.sqrt(x_sq + y_sq[j])

    if np.size(dims) == 3:
        x_sq = (np.arange(dims[0]) - dims[0] // 2) ** 2
        y_sq = (np.arange(dims[1]) - dims[1] // 2) ** 2
        z_sq = (np.arange(dims[2]) - dims[2] // 2) ** 2
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


def test_find_support():
    shape_v = [100, 100]
    cenv = shape_v[0] // 2
    r = 20
    a = np.zeros(shape_v)
    a[cenv - r : cenv + r, cenv - r : cenv + r] = 1.0
    sw_sigma = 0.50
    sw_threshold = 0.05

    new_sup_index = find_support(a, sw_sigma, sw_threshold)
    new_sup = np.zeros_like(a)
    new_sup[new_sup_index] = 1
    # the area of new support becomes larger
    assert np.sum(new_sup) == 1760


def make_synthetic_data():
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
    a[shapev[0] // 2 - r : shapev[0] // 2 + r, shapev[1] // 2 - r : shapev[1] // 2 + r] = 1
    diff_v = np.abs(np.fft.fftn(a)) / np.sqrt(np.size(a))
    return a, diff_v


def test_pi_modulus():
    a, diff_v = make_synthetic_data()
    a_new = pi_modulus(a, diff_v)
    assert_array_almost_equal(np.abs(a_new), a)


def test_cal_diff_error():
    a, diff_v = make_synthetic_data()
    result = cal_diff_error(a, diff_v)
    assert_equal(np.sum(result), 0)


def cal_support(func):
    def inner(*args):
        return func(*args)

    return inner


def _box_support_area(sup_radius, shape_v):
    sup = generate_box_support(sup_radius, shape_v)
    new_sup = sup[sup != 0]
    assert_array_equal(new_sup.shape, (2 * sup_radius) ** len(shape_v))


def _disk_support_area(sup_radius, shape_v):
    sup = generate_disk_support(sup_radius, shape_v)
    new_sup = sup[sup != 0]
    assert new_sup.size < (2 * sup_radius) ** len(shape_v)


@pytest.mark.parametrize("v", [(100, 100), (100, 100, 100)])
def test_support(v):
    sup_radius = 20
    _box_support_area(sup_radius, v)
    _disk_support_area(sup_radius, v)


def test_recon():
    a, diff_v = make_synthetic_data()
    total_n = 10
    sup_radius = 20

    # inital phase and support
    init_phase = generate_random_phase_field(diff_v)
    sup = generate_box_support(sup_radius, diff_v.shape)
    # run reconstruction
    outv1, error_dict = cdi_recon(diff_v, init_phase, sup, sw_flag=False, n_iterations=total_n, sw_step=2)
    outv1 = np.abs(outv1)

    outv2, error_dict = cdi_recon(
        diff_v, init_phase, sup, pi_modulus_flag="Real", sw_flag=True, n_iterations=total_n, sw_step=2
    )
    outv2 = np.abs(outv2)
    # compare the area of supports
    assert_array_equal(outv1.shape, outv2.shape)


def test_cdi_plotter():
    a, diff_v = make_synthetic_data()
    total_n = 10
    sup_radius = 20

    # inital phase and support
    init_phase = generate_random_phase_field(diff_v)
    sup = generate_box_support(sup_radius, diff_v.shape)

    # assign wrong plot_function
    with pytest.raises(TypeError):
        outv, error_d = cdi_recon(
            diff_v, init_phase, sup, sw_flag=True, n_iterations=total_n, sw_step=2, cb_function=10
        )
