import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from skxray.core.accumulators.histogram import Histogram


def _1d_histogram_tester(binlowhighs, *coords, weights):
    h = Histogram(binlowhighs)
    h.fill(coords, weights=weights)
    ynp = np.histogram(x, h.edges[0], weights=w)[0]
    assert_array_almost_equal(ynp, h.values)


def test_1d_histogram():
    binlowhigh = [10, 0, 10.01]
    x = np.random.random(1000000)*40
    xi = x.astype(int)
    w = np.linspace(1, 10, len(x))
    wi = w.copy()
    vals = [
        [binlowhigh, x, w],
        [binlowhigh, x, 1],
        [binlowhigh, xi, wi],
        [binlowhigh, xi, 1],
        [binlowhigh, x, wi],
        [binlowhigh, xi, w],
    ]
    for binlowhigh, x, w in vals:
        yield _1d_histogram_tester, binlowhigh, x, w


def test_argtypes():
    h = Histogram([10, 0, 10.01])
    h.fill(x, weights=w)
    h.fill(x, weights=wi)
    h.fill(xi, weights=w)
    h.fill(xi, weights=wi)
    h.fill(x)
    h.fill(xi)
    h.fill(x, weights=1.0)
    return


if __name__ == '__main__':
    test_1d_histogram()
