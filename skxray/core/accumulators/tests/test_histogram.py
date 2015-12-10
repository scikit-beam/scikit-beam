import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from skxray.core.accumulators.histogram import Histogram


def _1d_histogram_tester(binlowhighs, coords, weights=1):
    h = Histogram(binlowhighs)
    h.fill(coords, weights=weights)
    if np.isscalar(weights):
        ynp = np.histogram(coords, h.edges[0])[0]
    else:
        ynp = np.histogram(coords, h.edges[0], weights=weights)[0]

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


if __name__ == '__main__':
    test_1d_histogram()
