from skxray.core.correlation.corr import process_wrapper as cyprocess
from skxray.core.correlation.correlation import _process as pyprocess
from skxray.core.correlation.tests.test_correlation import FakeStack
from skxray.core.correlation import multi_tau_auto_corr
import numpy as np
import time as ttime

if __name__ == "__main__":
    processing_funcs = [pyprocess, cyprocess]

    num_levels = 4
    num_bufs = 4  # must be even
    xdim = 512
    ydim = 512
    stack_size = 1000
    img_stack = FakeStack(ref_img=np.zeros((xdim, ydim), dtype=int),
                          maxlen=stack_size)

    rois = np.zeros_like(img_stack[0])
    # make sure that the ROIs can be any integers greater than 1. They do not
    # have to start at 1 and be continuous
    # rois[:] = 1
    rois[0:xdim//10, 0:ydim//10] = 5
    rois[xdim//10:xdim//5, ydim//10:ydim//5] = 3

    for proc in processing_funcs:
        t0 = ttime.time()
        g2, lag_steps = multi_tau_auto_corr(num_levels, num_bufs, rois,
                                            img_stack,
                                            processing_func=proc)
        t1 = ttime.time()
        print("%s took %s seconds" % (proc, t1-t0))
