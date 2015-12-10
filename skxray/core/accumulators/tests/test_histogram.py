import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from skxray.core.accumulators.histogram import Histogram


def _1d_histogram_tester(binlowhighs, x, weights=1):
    h = Histogram(binlowhighs)
    h.fill(x, weights=weights)
    if np.isscalar(weights):
        ynp = np.histogram(x, h.edges[0])[0]
    else:
        ynp = np.histogram(x, h.edges[0], weights=weights)[0]

    assert_array_almost_equal(ynp, h.values)


def test_1d_histogram():
    binlowhigh = [10, 0, 10.01]
    xf = np.random.random(1000000)*40
    xi = xf.astype(int)
    wf = np.linspace(1, 10, len(xf))
    wi = wf.copy()
    vals = [
        [binlowhigh, xf, wf],
        [binlowhigh, xf, 1],
        [binlowhigh, xi, wi],
        [binlowhigh, xi, 1],
        [binlowhigh, xf, wi],
        [binlowhigh, xi, wf],
    ]
    for binlowhigh, x, w in vals:
        yield _1d_histogram_tester, binlowhigh, x, w



def _2d_histogram_tester(binlowhighs, x, y, weights=1):
    h = Histogram(*binlowhighs)
    h.fill(x, y, weights=weights)
    if np.isscalar(weights):
        ynp = np.histogram2d(x, y, bins=h.edges)[0]
    else:
        ynp = np.histogram2d(x, y, bins=h.edges, weights=weights)[0]

    assert_array_almost_equal(ynp, h.values)


def test_2d_histogram():
    ten = [10, 0, 10.01]
    nine = [9, 0, 9.01]
    xf = np.random.random(1000000)*40
    yf = np.random.random(1000000)*40
    xi = xf.astype(int)
    yi = yf.astype(int)
    wf = np.linspace(1, 10, len(xf))
    wi = wf.copy()
    vals = [
        [[ten, ten], xf, yf, wf],
        [[ten, nine], xf, yf, 1],
        [[ten, ten], xi, yi, wi],
        [[ten, ten], xi, yi, 1],
        [[ten, nine], xf, yf, wi],
        [[ten, nine], xi, yi, wf],
    ]
    for binlowhigh, x, y, w in vals:
        yield _2d_histogram_tester, binlowhigh, x, y, w


if __name__ == '__main__':
    test_1d_histogram()
    test_2d_histogram()

#TODO do a better job sampling the variable space
