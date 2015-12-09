import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from skxray.core.accumulators.histogram import hist1d

x = np.random.random(1000000)*40
xi = x.astype(int)
w = np.linspace(1, 10, len(x))
wi = w.astype(int)

def test_histfloat():
    h = hist1d(10, 0, 10.01)
    h.fill(x, w)
    ynp = np.histogram(x, h.nbinx,
            range=(h.xaxis.low, h.xaxis.high), weights=w)[0]
    assert_array_almost_equal(ynp, h.data)
    return


def test_histint():
    h = hist1d(10, 0, 10.01)
    h.fill(xi, wi)
    ynp = np.histogram(xi, h.nbinx,
            range=(h.xaxis.low, h.xaxis.high), weights=wi)[0]
    assert_array_equal(ynp, h.data)
    return


if __name__ == '__main__':
    test_histfloat()
    test_histint()
