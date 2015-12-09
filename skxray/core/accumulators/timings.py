import timeit
import time
import htest
import numpy as np

h = htest.hist1d(10, 0, 10);
x = np.random.random(1000000)*40;
w = np.ones_like(x)
xi = x.astype(int)
wi = np.ones_like(xi)
gg = globals()

def timethis(stmt):
    return np.mean(timeit.repeat(stmt, number=10, repeat=5, globals=gg))

def histfromzero(h, fncname, x, w):
    h.data[:] = 0
    getattr(h, fncname)(x, w)
    return h.data.copy()

print("Timing float")
print("Cython:", timethis('h.fillcy(x, w)'))
print("Cython with call:", timethis('h.fillcywithcall(x, w)'))
print("Numpy", timethis('h.fillnp(x, w)'))
# print("Python Looping", timethis('h.fill(x, w)'))

hnp = histfromzero(h, 'fillnp', x, w)
assert np.array_equal(hnp, histfromzero(h, 'fillcy', x, w))
assert np.array_equal(hnp, histfromzero(h, 'fillcywithcall', x, w))

hnp = histfromzero(h, 'fillnp', xi, wi)
assert np.array_equal(hnp, histfromzero(h, 'fillcy', xi, wi))
assert np.array_equal(hnp, histfromzero(h, 'fillcywithcall', xi, wi))


print()

print("Timing int")
print("Cython:", timethis('h.fillcy(xi, wi)'))
print("Cython with call:", timethis('h.fillcywithcall(xi, wi)'))
print("Numpy", timethis('h.fillnp(xi, wi)'))
# print("Python Looping", timethis('h.fill(xi, wi)'))
