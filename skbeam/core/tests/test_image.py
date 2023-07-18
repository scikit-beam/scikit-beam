from __future__ import absolute_import, division, print_function

import numpy as np
import numpy.random
import pytest
import skimage.draw as skd
from numpy.testing import assert_array_almost_equal, assert_equal, assert_raises
from scipy.ndimage import binary_dilation

import skbeam.core.image as nimage
from skbeam.core import roi


def _gen_test_find_ring_center_acorr_1D():
    vals = []
    for x in [110, 150, 190]:
        for y in [110, 150, 190]:
            vals.append((x, y))
    return vals


param_test_find_ring_center_acorr_1D = _gen_test_find_ring_center_acorr_1D()


@pytest.mark.parametrize("x, y", param_test_find_ring_center_acorr_1D)
def test_find_ring_center_acorr_1D(x, y):
    _helper_find_rings(nimage.find_ring_center_acorr_1D, (x, y), [10, 25, 50])


def _helper_find_rings(proc_method, center, radii_list):
    x, y = center
    image_size = (256, 265)
    numpy.random.seed(42)
    noise = np.random.rand(*image_size)
    tt = np.zeros(image_size)
    for r in radii_list:
        rr, cc = skd.circle_perimeter(x, y, r)
        tt[rr, cc] = 1
    tt = binary_dilation(tt, structure=np.ones((3, 3))).astype(float) * 100

    tt = tt + noise
    res = proc_method(tt)
    assert_equal(res, center)


def test_construct_circ_avg_image():
    # need to test with center and dims and without
    # and test anisotropic pixels
    image = np.zeros((12, 12))
    calib_center = (2, 2)
    inner_radius = 1

    edges = roi.ring_edges(inner_radius, width=1, spacing=1, num_rings=2)
    labels = roi.rings(edges, calib_center, image.shape)
    image[labels == 1] = 10
    image[labels == 2] = 10

    bin_cen, ring_avg = roi.circular_average(image, calib_center, nx=6)

    # check that the beam center and dims yield the circavg in the right place
    cimg = nimage.construct_circ_avg_image(bin_cen, ring_avg, dims=image.shape, center=calib_center)

    assert_array_almost_equal(
        cimg[2],
        np.array(
            [
                5.0103283,
                6.15384615,
                6.15384615,
                6.15384615,
                5.0103283,
                3.79296498,
                2.19422113,
                0.51063356,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
    )

    # check that the default values center in the same place
    cimg2 = nimage.construct_circ_avg_image(bin_cen, ring_avg)

    assert_array_almost_equal(
        cimg2[12],
        np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.51063356,
                2.19422113,
                3.79296498,
                5.0103283,
                6.15384615,
                6.15384615,
                6.15384615,
                5.0103283,
                3.79296498,
                2.19422113,
                0.51063356,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
    )

    # check that anisotropic pixels are treated properly
    cimg3 = nimage.construct_circ_avg_image(bin_cen, ring_avg, dims=image.shape, pixel_size=(2, 1))

    assert_array_almost_equal(
        cimg3[5],
        np.array(
            [
                0.0,
                1.16761618,
                2.80022015,
                4.16720388,
                5.250422,
                6.08400137,
                6.08400137,
                5.250422,
                4.16720388,
                2.80022015,
                1.16761618,
                0.0,
            ]
        ),
    )

    with assert_raises(ValueError):
        nimage.construct_circ_avg_image(bin_cen, ring_avg, center=calib_center, pixel_size=(2, 1))


def test_construct_rphi_avg_image():
    Nr, Na = 40, 40
    Nx, Ny = 44, 44

    angles1 = np.linspace(0, 2 * np.pi - 0.1, Na)
    angles2 = np.linspace(0, 2 * np.pi, Na, endpoint=False)
    angles3 = np.linspace(0, 2 * np.pi + 0.1, Na)
    radii = np.linspace(0, 10, Nr)

    ANGLES1, RADII = np.meshgrid(angles1, radii)
    ANGLES2, RADII = np.meshgrid(angles2, radii)
    ANGLES3, RADII = np.meshgrid(angles3, radii)
    Z1 = np.cos(ANGLES1 * 2) ** 2 * RADII**2
    Z2 = np.cos(ANGLES2 * 2) ** 2 * RADII**2
    Z3 = np.cos(ANGLES3 * 2) ** 2 * RADII**2

    mask = np.ones_like(RADII)
    mask[1:8] = 0
    mask[:, 1:8] = 0
    mask[:, 5:12] = 0
    mask[7:18] = 0

    # only try mask on one, rest is redundant
    Z_masked = Z2 * mask
    angles_masked = angles2

    # ony try one version for anisotropy
    Z_anis = Z2
    angles_anis = angles2

    shape = np.array([Nx, Ny])

    # version  1
    Zproj1 = nimage.construct_rphi_avg_image(radii, angles1, Z1, shape=shape)
    Zproj2 = nimage.construct_rphi_avg_image(radii, angles2, Z2, shape=shape)
    with assert_raises(ValueError):
        nimage.construct_rphi_avg_image(radii, angles3, Z3, shape=shape)

    # try masked versions
    Zproj_masked = nimage.construct_rphi_avg_image(radii, angles_masked, Z_masked, shape=shape, mask=mask)

    # anisotropy version
    Zproj_anis = nimage.construct_rphi_avg_image(radii, angles_anis, Z_anis, shape=shape, pixel_size=(1, 0.5))

    assert_array_almost_equal(Zproj1[::8, 22], np.array([np.nan, np.nan, 28.9543923, 5.44672546, np.nan, np.nan]))

    assert_array_almost_equal(Zproj2[::8, 22], np.array([np.nan, np.nan, 28.834469, 5.465199, np.nan, np.nan]))

    assert_array_almost_equal(Zproj_masked[::8, 22], np.array([np.nan, np.nan, 28.834469, np.nan, np.nan, np.nan]))

    assert_array_almost_equal(Zproj_anis[::8, 22], np.array([np.nan, np.nan, 29.491395, 5.939956, np.nan, np.nan]))
