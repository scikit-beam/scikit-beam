from skxray.core.correlation import multi_tau_auto_corr
from skxray.core.accumulators.correlation import MultiTauCorrelation
import numpy as np


def test_against_reference_implementation():
    num_levels = 4
    num_bufs = 4  # must be even
    xdim = 256
    ydim = 512
    stack_size = 20
    # img_stack = np.zeros((stack_size, xdim, ydim), dtype=int)
    img_stack = np.random.randint(1, 10, ((stack_size, xdim, ydim)))
    rois = np.zeros_like(img_stack[0])
    # make sure that the ROIs can be any integers greater than 1. They do not
    # have to start at 1 and be continuous
    rois[0:xdim//10, 0:ydim//10] = 5
    rois[xdim//10:xdim//5, ydim//10:ydim//5] = 3
    # run the correlation with the reference implementation
    ret = multi_tau_auto_corr(num_levels, num_bufs, rois, img_stack)
    g2, lag_steps, G, buf, past_intensity_norm, future_intensity_norm = ret
    # run the correlation with the accumulator version
    mt = MultiTauCorrelation(num_levels, num_bufs, rois)
    for img in img_stack:
        mt.process(img)

    # compare the results
    assert np.all(mt.g2 == g2)
    assert np.all(mt.lag_steps == lag_steps)
    #TODO Figure out why mt._future_intensity_norm is different from
    # future_intensity_norm
    raise
