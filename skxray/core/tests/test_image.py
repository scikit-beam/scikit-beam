from __future__ import absolute_import, division, print_function
import numpy as np
import numpy.random
from nose.tools import assert_equal
import skimage.draw as skd
from scipy.ndimage.morphology import binary_dilation

import skxray.core.image as nimage


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


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
