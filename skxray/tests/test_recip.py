from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
from nose.tools import raises

from skxray.testing.decorators import known_fail_if
import numpy.testing as npt
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)

from nose.tools import assert_equal, assert_true, raises
from skxray import recip


@known_fail_if(six.PY3)
def test_process_to_q():
    detector_size = (256, 256)
    pixel_size = (0.0135*8, 0.0135*8)
    calibrated_center = (256/2.0, 256/2.0)
    dist_sample = 355.0

    energy = 640  # (  in eV)
    # HC_OVER_E to convert from Energy to wavelength (Lambda)
    hc_over_e = 12398.4
    wavelength = hc_over_e / energy  # (Angstrom )

    ub_mat = np.array([[-0.01231028454, 0.7405370482, 0.06323870032],
                       [0.4450897473, 0.04166852402, -0.9509449389],
                       [-0.7449130975, 0.01265920962, -0.5692399963]])

    setting_angles = np.array([[40., 15., 30., 25., 10., 5.],
                              [90., 60., 0., 30., 10., 5.]])
    # delta=40, theta=15, chi = 90, phi = 30, mu = 10.0, gamma=5.0
    pdict = {}
    pdict['setting_angles'] = setting_angles
    pdict['detector_size'] = detector_size
    pdict['pixel_size'] = pixel_size
    pdict['calibrated_center'] = calibrated_center
    pdict['dist_sample'] = dist_sample
    pdict['wavelength'] = wavelength
    pdict['ub'] = ub_mat
    # ensure invalid entries for frame_mode actually fail

    # todo test frame_modes 1, 2, and 3
    # test that the values are coming back as expected for frame_mode=4
    hkl = recip.process_to_q(**pdict)

    # Known HKL values for the given six angles)
    # each entry in list is (pixel_number, known hkl value)
    known_hkl = [(32896, np.array([-0.15471196, 0.19673939, -0.11440936])),
                 (98432, np.array([0.10205953,  0.45624416, -0.27200778]))]

    for pixel, kn_hkl in known_hkl:
        npt.assert_array_almost_equal(hkl[pixel], kn_hkl, decimal=8)

    # smoketest the frame_mode variable
    pass_list = recip.process_to_q.frame_mode
    pass_list.append(None)
    for passes in pass_list:
        recip.process_to_q(frame_mode=passes, **pdict)


@raises(KeyError)
def _process_to_q_exception(param_dict, frame_mode):
    recip.process_to_q(frame_mode=frame_mode, **param_dict)


def test_frame_mode_fail():
    detector_size = (256, 256)
    pixel_size = (0.0135*8, 0.0135*8)
    calibrated_center = (256/2.0, 256/2.0)
    dist_sample = 355.0

    energy = 640  # (  in eV)
    # HC_OVER_E to convert from Energy to wavelength (Lambda)
    hc_over_e = 12398.4
    wavelength = hc_over_e / energy  # (Angstrom )

    ub_mat = np.array([[-0.01231028454, 0.7405370482, 0.06323870032],
                       [0.4450897473, 0.04166852402, -0.9509449389],
                       [-0.7449130975, 0.01265920962, -0.5692399963]])

    setting_angles = np.array([[40., 15., 30., 25., 10., 5.],
                              [90., 60., 0., 30., 10., 5.]])
    # delta=40, theta=15, chi = 90, phi = 30, mu = 10.0, gamma=5.0
    pdict = {}
    pdict['setting_angles'] = setting_angles
    pdict['detector_size'] = detector_size
    pdict['pixel_size'] = pixel_size
    pdict['calibrated_center'] = calibrated_center
    pdict['dist_sample'] = dist_sample
    pdict['wavelength'] = wavelength
    pdict['ub'] = ub_mat

    for fails in [0, 5, 'cat']:
        yield _process_to_q_exception, pdict, fails


def test_hkl_to_q():
    b = np.array([[-4, -3, -2],
                  [-1, 0, 1],
                  [2, 3, 4],
                  [6, 9, 10]])

    b_norm = np.array([5.38516481, 1.41421356, 5.38516481,
                       14.73091986])

    npt.assert_array_almost_equal(b_norm, recip.hkl_to_q(b))


def test_q_rings():
    #xx, yy = np.mgrid[:15, :12]
    calibrated_center = (0.5, 0.5)
    img_dim = (15, 12)
    #circle = (xx - 0.5) ** 2 + (yy - 0.5) ** 2
    #q_val = np.ravel(circle)

    first_q = 2.5
    delta_q = 2.5
    num_qs = 20  # number of Q rings

    (q_inds, q_ring_val, num_pixels, all_pixels,
     pixel_list) = recip.q_no_step_val(img_dim, calibrated_center, num_qs,
               first_q, delta_q)

    #q_ring_val, q_inds = recip.q_no_step_val(q_val, num_qs, first_q,
    #                                         delta_q, q_val)
    #(q_inds, q_ring_val, num_pixels, pixel_list) = recip.q_rings(num_qs,
    #                                                             circle.shape,
    #                                                            q_ring_val,
    #                                                             q_inds)

    q_inds_m = np.array([[0, 0, 1, 2, 5, 8, 12, 17, 0, 0, 0, 0],
                        [0, 0, 1, 2, 5, 8, 12, 17, 0, 0, 0, 0],
                        [1, 1, 1, 3, 5, 9, 13, 17, 0, 0, 0, 0],
                        [2, 2, 3,  5, 7, 10, 14, 19, 0, 0, 0, 0],
                        [5, 5, 5,  7, 9, 13, 17, 0, 0, 0, 0, 0],
                        [8, 8, 9, 10, 13, 16, 20, 0, 0, 0, 0, 0],
                        [12, 12, 13, 14, 17, 20, 0, 0, 0, 0, 0, 0],
                        [17, 17, 17, 19, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    q_ring_val_m = np.array([[2.5, 5.],
                             [5., 7.5],
                             [7.5, 10.],
                             [10., 12.5],
                             [12.5, 15.],
                             [15., 17.5],
                             [17.5, 20.],
                             [20., 22.5],
                             [22.5, 25.],
                             [25., 27.5],
                             [27.5, 30.],
                             [30., 32.5],
                             [32.5, 35.],
                             [35., 37.5],
                             [37.5, 40.],
                             [40., 42.5],
                             [42.5, 45.],
                             [45., 47.5],
                             [47.5, 50.],
                             [50., 52.5]])

    num_pixels_m = np.array(([5, 4, 2, 0, 7, 0, 2, 4, 3, 2, 0, 4, 4, 2,
                              0, 1, 8, 0, 2, 2]))

    pixel_list_m = np.array([30,  45, 60, 75, 90, 105, 31, 46, 61,
                             76, 91, 106, 2, 17,  32, 47, 62, 77,
                             92, 107, 3, 18, 33, 48, 63, 78, 93, 108,
                             4, 19, 34, 49, 64, 79, 94,  5, 20, 35, 50,
                             65,  80, 95, 6, 21, 36, 51, 66, 81, 7, 22,
                             37, 52])

    assert_array_almost_equal(q_ring_val_m, q_ring_val)
    assert_array_equal(num_pixels, num_pixels_m)
    assert_array_equal(q_inds, np.ravel(q_inds_m))
    assert_array_equal(pixel_list, pixel_list_m)

    # using a step for the Q rings
    (qstep_inds, qstep_ring_val, numstep_pixels, allstep_pixels,
     pixelstep_list) = recip.q_step_val(img_dim, calibrated_center, num_qs,
               first_q, delta_q, 0.5)

    #(qstep_ring_val, qstep_inds) = recip.q_step_val(q_val, num_qs,
    #                                               first_q, delta_q, 0.5)
    #(qstep_inds, qstep_ring_val,
    # numstep_pixels, pixelstep_list) = recip.q_rings(num_qs, circle.shape,
    #                                 qstep_ring_val, qstep_inds)

    qstep_inds_m = np.array([[0, 0, 1, 2, 4, 7, 10, 14, 19, 0, 0, 0],
                            [0, 0, 1, 2, 4, 7, 10, 14, 19, 0, 0, 0],
                            [1, 1, 1, 3, 5, 7, 11, 15, 19, 0, 0, 0],
                            [2, 2, 3, 4, 6, 9, 12, 16, 0, 0, 0, 0],
                            [4, 4, 5, 6, 8, 11, 14, 18, 0, 0, 0, 0],
                            [7, 7, 7, 9, 11, 13, 17, 0, 0, 0, 0, 0],
                            [10, 10, 11, 12, 14, 17, 20, 0, 0, 0, 0, 0],
                            [14, 14, 15, 16, 18, 0, 0, 0, 0, 0, 0, 0],
                            [19, 19, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    numstep_pixels_m = np.array([5, 4, 2, 5, 2, 2, 6, 1, 2, 4, 4, 2, 1,
                                 6, 2, 2, 2, 2, 6, 1])

    qstep_ring_val_m = np.array([[2.5, 5.],
                                [5.5, 8.],
                                [8.5, 11.],
                                [11.5, 14.],
                                [14.5, 17.],
                                [17.5, 20.],
                                [20.5, 23.],
                                [23.5, 26.],
                                [26.5, 29.],
                                [29.5, 32.],
                                [32.5, 35.],
                                [35.5, 38.],
                                [38.5, 41.],
                                [41.5, 44.],
                                [44.5, 47.],
                                [47.5, 50.],
                                [50.5, 53.],
                                [53.5, 56.],
                                [56.5, 59.],
                                [59.5, 62.]])

    pixelstep_list_m = np.array([30, 45, 60, 75, 90, 105, 120, 31,
                                 46, 61, 76, 91, 106, 121,  2, 17,
                                 32, 47, 62, 77, 92, 107, 122, 3,
                                 18, 33, 48, 63, 78, 93, 108, 4,
                                 19, 34, 49, 64,  79, 94, 109, 5,
                                 20, 35, 50, 65, 80, 95, 6, 21, 36,
                                 51, 66, 81, 96,  7, 22, 37, 52, 67,
                                 8, 23, 38])

    assert_almost_equal(qstep_ring_val, qstep_ring_val_m)
    assert_array_equal(numstep_pixels, numstep_pixels_m)
    assert_array_equal(qstep_inds, np.ravel(qstep_inds_m))

    assert_array_equal(pixelstep_list, pixelstep_list_m)

    num_qs = 8
    (qd_inds, qd_ring_val, numd_pixels, alld_pixels,
     pixeld_list) = recip.q_step_val(img_dim, calibrated_center, num_qs,
               first_q, delta_q,  0.4, 0.2, 0.5, 0.4, 0.0, 0.6, 0.2)


    #(qd_ring_val, qd_inds) = recip.q_step_val(q_val, num_qs,
    #                                          first_q, delta_q, 0.4, 0.2,
                                              #0.5, 0.4, 0.0, 0.6, 0.2)

    #(qd_inds, qd_ring_val, numd_pixels,
    # pixeld_list) = recip.q_rings(num_qs, circle.shape, qd_ring_val, qd_inds)

    qd_ring_val_m = ([[2.5, 5.],
                     [5.4, 7.9],
                     [8.1, 10.6],
                     [11.1, 13.6],
                     [14., 16.5],
                     [16.5, 19.],
                     [19.6, 22.1],
                     [22.3, 24.8]])

    numd_pixels_m = np.array([5, 4, 2, 5, 2, 2, 4, 3])

    pixeld_list_m = np.array([30, 45, 60, 75, 31, 46, 61, 76, 2, 17, 32, 47,
                              62, 77,  3, 18, 33, 48, 63, 4, 19, 34, 49, 64,
                              5, 20, 35])

    qd_inds_m = np.array([[0, 0, 1, 2, 4, 7, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 2, 4, 7, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 3, 5, 8, 0, 0, 0, 0, 0, 0],
                         [2, 2, 3, 4, 6, 0, 0, 0, 0, 0, 0, 0],
                         [4, 4, 5, 6, 8, 0, 0, 0, 0, 0, 0, 0],
                         [7, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    assert_array_equal(qd_inds, np.ravel(qd_inds_m))
    assert_array_equal(qd_ring_val, qd_ring_val_m)
    assert_array_equal(numd_pixels, numd_pixels_m)
    assert_array_equal(pixeld_list, pixeld_list_m)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
