from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import nsls2.recip as recip
import numpy.testing as npt
from numpy.testing.noseclasses import KnownFailureTest
import six


def test_process_to_q():
    if six.PY3:
        return
    detector_size = (256, 256)
    pixel_size = (0.0135*8, 0.0135*8)
    calibrated_center = (256/2.0, 256/2.0)
    dist_sample = 355.0

    energy = 640  # (  in eV)
    # HC_OVER_E to convert from Energy to wavelength (Lambda)
    hc_over_e = 12398.4
    wavelength = hc_over_e / energy  # (Angstrom )

    ub_mat = np.matrix([[-0.01231028454, 0.7405370482, 0.06323870032],
                       [0.4450897473, 0.04166852402, -0.9509449389],
                       [-0.7449130975, 0.01265920962, -0.5692399963]])

    setting_angles = np.matrix([[40., 15., 30., 25., 10., 5.],
                              [90., 60., 0., 30., 10., 5.]])
    # delta=40, theta=15, chi = 90, phi = 30, mu = 10.0, gamma=5.0

    tot_set = recip.process_to_q(setting_angles, detector_size, pixel_size, calibrated_center,
                                dist_sample, wavelength, ub_mat)

    #  Known HKL values for the given six angles)
    hkl1 = np.matrix([[-0.15471196, 0.19673939, -0.11440936]])
    hkl2 = np.matrix([[0.10205953,  0.45624416, -0.27200778]])

    # New HKL values obtained from the process_to_q

    n_hkl1 = np.around(np.matrix([[tot_set[32896, 0],
                    tot_set[32896, 1], tot_set[32896, 2]]]), decimals=8)
    n_hkl2 = np.around(np.matrix([[tot_set[98432, 0],
                    tot_set[98432, 1], tot_set[98432, 2]]]), decimals=8)

    # check the values are as expected
    npt.assert_array_equal(hkl1, n_hkl1)
    npt.assert_array_equal(hkl2, n_hkl2)


def test_process_grid():
    if six.PY3:
        return
    size = 4
    sigma = 0.1
    q_max = np.array([1.0, 1.0, 1.0])
    q_min = np.array([-1.0, -1.0, -1.0])
    dqn = np.array([size, size, size])

    grid = np.mgrid[0:dqn[0], 0:dqn[1], 0:dqn[2]]
    r = (q_max - q_min) / dqn

    X = grid[0] * r[0] + q_min[0]
    Y = grid[1] * r[1] + q_min[1]
    Z = grid[2] * r[2] + q_min[2]

    out = np.zeros((size, size, size))

    out = np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma**2))

    data = np.array([np.ravel(X),
                     np.ravel(Y),
                     np.ravel(Z)])
    data = data.T

    grid_data, grid_occu, grid_std, grid_out = recip.process_grid(data, out, q_max, q_min, dqn)
    
    # Values that have to go to the gridder
    databack = np.ravel(out)

    # Values from the gridder
    grid_databack = np.ravel(grid_data)

    # check the values are as expected
    npt.assert_array_almost_equal(grid_databack, databack)
