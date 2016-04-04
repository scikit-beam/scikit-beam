
if __name__ == "__main__":
    import timeit
    import numpy as np
    from skbeam.core.accumulators.histogram import Histogram
    h = Histogram((10, 0, 10.1), (7, 0, 7.1))
    x = np.random.random(1000000)*40
    y = np.random.random(1000000)*10
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

    print("Timing h.fill", timethis('h.fill(x, y, weights=w)'))

    h._always_use_fillnd = True
    print("Timing h.fill with _always_use_fillnd",
          timethis('h.fill(x, y, weights=w)'))
