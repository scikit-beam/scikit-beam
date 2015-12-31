from skxray.core.correlation.cyprocess import cyprocess
from skxray.core.correlation.pyprocess import pyprocess
from skxray.core.correlation.tests.test_correlation import FakeStack
from skxray.core.correlation import multi_tau_auto_corr
from skxray.core import roi
import itertools
import numpy as np
import time as ttime
import pandas as pd


def make_ring_array(xdim, ydim, fraction_roi, num_rois):
    radius = np.sqrt(fraction_roi * xdim * ydim / np.pi)
    edges = np.linspace(0, radius, num_rois+1)
    ring_pairs = [(x0, x1) for x0, x1 in zip(edges, edges[1:])]
    return roi.rings(edges=ring_pairs, center=(xdim//2, ydim//2), shape=(xdim, ydim))


def timer(num_levels, num_bufs, rois, img_stack, func):
    t0 = ttime.time()
    g2, lag_steps = multi_tau_auto_corr(num_levels, num_bufs, rois,
                                        img_stack, func)
    return g2, lag_steps, ttime.time() - t0


if __name__ == "__main__":
    processing_funcs = {'python': pyprocess, 'cython': cyprocess}
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
        img_stack = FakeStack(ref_img=np.zeros((x, x), dtype=int),
                              maxlen=z)

        rois = make_ring_array(x, x, occupancy, nroi)
        pyg2, pylag, pytime = timer(num_levels, num_bufs, rois, img_stack,
                                    pyprocess)
        cyg2, cylag, cytime = timer(num_levels, num_bufs, rois, img_stack,
                                    cyprocess)

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
