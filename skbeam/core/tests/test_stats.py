from skbeam.core.stats import statistics_1D, poissonize
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal


def test_statistics_1D():
    # set up simple data
    x = np.linspace(0, 1, 100)
    y = np.arange(100)
    nx = 10
    # make call
    edges, val = statistics_1D(x, y, nx=nx)
    # check that values are as expected
    assert_array_almost_equal(edges,
                              np.linspace(0, 1, nx + 1, endpoint=True))
    assert_array_almost_equal(val,
                              np.sum(y.reshape(nx, -1), axis=1)/10.)


def test_poissonize():
    # choose some seed
    seed = 41301823
    np.random.seed(seed)
    mu = 1000
    x = np.ones(10)*mu
    v1 = poissonize(x)
    v2 = poissonize(x, Navg=10)

    v1avg = np.average(v1)
    v2avg = np.average(v2)
    assert_almost_equal(v1avg, 988.0)
    assert_almost_equal(v2avg, 1006.4)
    assert_array_almost_equal(v1, np.array([908., 1019., 1006., 1002., 973.,
                                            987., 950., 988., 1027., 1020.]))

    assert_array_almost_equal(v2, np.array([1014.4, 994.3, 1015.5, 1015.4,
                                            995.7, 988.5, 1002., 1020.3,
                                            1011.4, 1006.5]))
