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
    detPixSizeX, detPixSizeY, detSizeX, detSizeY = 0.0135*8, 0.0135*8, 256, 256
    detX0, detY0, detDis, detAng = 256/2.0, 256/2.0, 355.0, 0.0

    energy = 640  # (  in eV)
    # HC_OVER_E to convert from Energy to wavelength (Lambda)
    hc_over_e = 12398.4
    waveLen = hc_over_e / energy  # (Angstrom )

    UBmat = np.matrix([[-0.01231028454, 0.7405370482, 0.06323870032],
                       [0.4450897473, 0.04166852402, -0.9509449389],
                       [-0.7449130975, 0.01265920962, -0.5692399963]])

    settingAngles = np.matrix([[40., 15., 30., 25., 10., 5.],
                              [90., 60., 0., 30., 10., 5.]])
    # delta=40, theta=15, chi = 90, phi = 30, mu = 10.0, gamma=5.0

    totSet = recip.process_to_q(settingAngles, detSizeX, detSizeY,
                                detPixSizeX, detPixSizeY, detX0, detY0,
                                detDis, waveLen, UBmat)
    
    #  Known HKL values for the given six angles)
    HKL1 = np.matrix([[-0.15471196, 0.19673939, -0.11440936]])
    HKL2 = np.matrix([[0.10205953,  0.45624416, -0.27200778]])

    # New HKL values obtained from the process_to_q

    N_HKL1 = np.around(np.matrix([[totSet[32896, 0],
                    totSet[32896, 1], totSet[32896, 2]]]), decimals=8)
    N_HKL2 = np.around(np.matrix([[totSet[98432, 0],
                    totSet[98432, 1], totSet[98432, 2]]]), decimals=8)

    # check the values are as expected
    npt.assert_array_equal(HKL1, N_HKL1)
    npt.assert_array_equal(HKL2, N_HKL2)


def test_process_grid():
    if six.PY3:
        return
    size = 4
    sigma = 0.1
    Qmax = np.array([1.0, 1.0, 1.0])
    Qmin = np.array([-1.0, -1.0, -1.0])
    dQN = np.array([size, size, size])

    grid = np.mgrid[0:dQN[0], 0:dQN[1], 0:dQN[2]]
    r = (Qmax - Qmin) / dQN

    X = grid[0] * r[0] + Qmin[0]
    Y = grid[1] * r[1] + Qmin[1]
    Z = grid[2] * r[2] + Qmin[2]

    out = np.zeros((size, size, size))

    out = np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma**2))

    data = np.array([np.ravel(X),
                     np.ravel(Y),
                     np.ravel(Z)])
    data = data.T


    gridData, gridOccu, gridStd, gridOut, emptNb, gridbins = recip.process_grid(data, out, Qmax, Qmin, dQN)
    
    # Values that have to go to the gridder
    Databack = np.ravel(out)

    # Values from the gridder
    gridDataback = np.ravel(gridData)

    # check the values are as expected
    npt.assert_array_almost_equal(gridDataback, Databack)
