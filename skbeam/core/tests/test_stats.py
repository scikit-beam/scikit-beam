import numpy as np
from numpy.testing import assert_array_almost_equal

from skbeam.core.stats import statistics_1D


def test_statistics_1D():
    # set up simple data
    x = np.linspace(0, 1, 100)
    y = np.arange(100)
    nx = 10
    # make call
    edges, val = statistics_1D(x, y, nx=nx)
    # check that values are as expected
    assert_array_almost_equal(edges, np.linspace(0, 1, nx + 1, endpoint=True))
    assert_array_almost_equal(val, np.sum(y.reshape(nx, -1), axis=1) / 10.0)
