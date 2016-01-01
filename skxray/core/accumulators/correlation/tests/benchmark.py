from skxray.core.correlation.cyprocess import cyprocess
from skxray.core.correlation.pyprocess import pyprocess
from skxray.core.correlation.tests.test_correlation import FakeStack
from skxray.core.accumulators.correlation import lazy_multi_tau as pytau
from skxray.core.accumulators.cycorrelation import lazy_multi_tau as cytau
from skxray.core import roi
from numpy.testing import assert_array_almost_equal, assert_array_equal
import itertools
import numpy as np
import time as ttime
import pandas as pd


def make_ring_array(xdim, ydim, fraction_roi, num_rois):
    radius = np.sqrt(fraction_roi * xdim * ydim / np.pi)
    edges = np.linspace(0, radius, num_rois+1)
    ring_pairs = [(x0, x1) for x0, x1 in zip(edges, edges[1:])]
    return roi.rings(edges=ring_pairs, center=(xdim//2, ydim//2), shape=(xdim, ydim))


def cycall(num_levels, num_bufs, rois, img_stack):
    t0 = ttime.time()
    state = None
    for img in img_stack:
        res = cytau(img, num_levels, num_bufs, rois, state)
        state = res.internal_state
    return res.g2, res.lag_steps, ttime.time() - t0


def pycall(num_levels, num_bufs, rois, img_stack):
    t0 = ttime.time()
    gen = pytau(img_stack, num_levels, num_bufs, rois)
    res = list(gen)[-1]
    return res.g2, res.lag_steps, ttime.time() - t0


if __name__ == "__main__":
    correlation_funcs = {'python': pytau, 'cython': cytau}
    xdim = [128, 512]
    zdim = [10, 100]
    fraction_roi = [.001, .01, .1, .5, 1]
    num_rois = [1, 10, 100, 1000]

    num_levels = 4
    num_bufs = 4  # must be even
    benchmarks = []
    # table = PrettyTable(['x', 'y', 'z', 'occupancy', 'nroi', 'pytime',
    #                      'cytime', 'cytime/pytime'])
    fname = 'bench.txt'
    f = open(fname, 'w')
    f.write('x z occupancy nroi pytime cytime pytime/cytime\n')
    for x, z, occupancy, nroi in itertools.product(
            xdim, zdim, fraction_roi, num_rois):
        # make the image stack
        img_stack = FakeStack(ref_img=np.zeros((x, x), dtype=float),
                              maxlen=z)

        rois = make_ring_array(x, x, occupancy, nroi)
        pyg2, pylag, pytime = pycall(num_levels, num_bufs, rois, img_stack)
        cyg2, cylag, cytime = cycall(num_levels, num_bufs, rois, img_stack)

        assert_array_almost_equal(pyg2, cyg2)
        assert_array_equal(pylag, cylag)
        # bench = [x, y, z, occupancy, nroi, np.round(pytime, decimals=5),
        #          np.round(cytime, decimals=5),
        #          np.round(cytime/pytime, decimals=5)]
        # table.add_row(bench)
        # print(table)
        print("%4g | %4g | %5g | %4g | %7g | %7g | %7g" % (
            x, z, occupancy, nroi,
            np.round(pytime, decimals=5),
            np.round(cytime, decimals=5),
            np.round(pytime/cytime, decimals=5)))
        f.write('%s %s %s %s %s %s %s\n' % (x, z, occupancy, nroi, pytime,
                                            cytime, pytime/cytime))
    f.close()
    # print(repr(benchmarks))
    df = pd.read_csv(fname, sep=' ')
    mean, std = df['pytime/cytime'].mean(), df['pytime/cytime'].std()
    print("cython is %4g (%4g) seconds faster than python" % (mean, std))
