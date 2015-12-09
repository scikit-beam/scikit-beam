import timeit
import time
import numpy as np
from skxray.core.accumulators.histogram import hist1d

h = hist1d(10, 0, 10.1);
x = np.random.random(1000000)*40
w = np.ones_like(x)
xi = x.astype(int)
xi = xi.astype(float)
wi = np.ones_like(xi)
gg = globals()

def timethis(stmt):
    return np.mean(timeit.repeat(stmt, number=10, repeat=5, globals=gg))

def histfromzero(h, fncname, x, w):
    h.data[:] = 0
    getattr(h, fncname)(x, w)
    return h.data.copy()

print("Timing float")
print("Cython with call:", timethis('h.fill(x, w)'))

hnp = np.histogram(x, h.nbinx, range=(h.xaxis.low, h.xaxis.high), weights=w)[0]
assert np.array_equal(hnp, histfromzero(h, 'fill', x, w))

hnp = np.histogram(xi, h.nbinx, range=(h.xaxis.low, h.xaxis.high), weights=wi)[0]
assert np.array_equal(hnp, histfromzero(h, 'fill', xi, wi))

print()

print("Timing int")
print("Cython:", timethis('h.fill(xi, wi)'))
