from __future__ import absolute_import, division, print_function
import numpy as np
from skbeam.core import roi
import numpy.random
import skimage.draw as skd
from scipy.ndimage.morphology import binary_dilation
import skbeam.core.image as nimage

from numpy.testing import assert_array_almost_equal

from nose.tools import assert_equal, assert_raises


def test_find_ring_center_acorr_1D():
    for x in [110, 150, 190]:
        for y in [110, 150, 190]:
            yield (_helper_find_rings,
                   nimage.find_ring_center_acorr_1D,
                   (x, y), [10, 25, 50])


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
    cimg = nimage.construct_circ_avg_image(bin_cen, ring_avg, dims=image.shape,
                                           center=calib_center)

    assert_array_almost_equal(cimg[2], np.array([5.0103283, 6.15384615,
                              6.15384615, 6.15384615, 5.0103283, 3.79296498,
                              2.19422113, 0.51063356, 0., 0., 0., 0.]))

    # check that the default values center in the same place
    cimg2 = nimage.construct_circ_avg_image(bin_cen, ring_avg)

    assert_array_almost_equal(cimg2[12], np.array([0., 0., 0., 0., 0.,
                              0., 0., 0.51063356, 2.19422113, 3.79296498,
                              5.0103283, 6.15384615, 6.15384615, 6.15384615,
                              5.0103283, 3.79296498, 2.19422113, 0.51063356,
                              0., 0., 0., 0., 0., 0., 0.]))

    # check that anisotropic pixels are treated properly
    cimg3 = nimage.construct_circ_avg_image(bin_cen, ring_avg,
                                            dims=image.shape,
                                            pixel_size=(2, 1))

    assert_array_almost_equal(cimg3[5], np.array([0., 1.16761618, 2.80022015,
                              4.16720388, 5.250422, 6.08400137, 6.08400137,
                              5.250422,  4.16720388, 2.80022015,
                              1.16761618,  0.]))

    with assert_raises(ValueError):
        nimage.construct_circ_avg_image(bin_cen, ring_avg, center=calib_center,
                                        pixel_size=(2, 1))


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
