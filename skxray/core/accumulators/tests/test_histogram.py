import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from skxray.core.accumulators.histogram import Histogram

x = np.random.random(1000000)*40
xi = x.astype(int)
w = np.linspace(1, 10, len(x))
wi = w.astype(int)

def test_histfloat():
    h = Histogram([10, 0, 10.01])
    h.fill(x, weights=w)
    ynp = np.histogram(x, h.edges[0], weights=w)[0]
    assert_array_almost_equal(ynp, h.values)
    h.reset()
    h.fill(x)
    ynp = np.histogram(x, h.edges[0])[0]
    assert_array_almost_equal(ynp, h.values)
    return


def test_histint():
    h = Histogram([10, 0, 10.01])
    h.fill(xi, weights=wi)
    ynp = np.histogram(xi, h.edges[0], weights=wi)[0]
    assert_array_equal(ynp, h.values)
    h.reset()
    h.fill(xi)
    ynp = np.histogram(xi, h.edges[0])[0]
    assert_array_equal(ynp, h.values)
    return


if __name__ == '__main__':
    test_histfloat()
    test_histint()
