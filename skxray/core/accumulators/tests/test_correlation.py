from skxray.core.correlation.correlation import multi_tau_auto_corr
from skxray.core.accumulators.correlation import (MultiTauCorrelation,
                                                  intermediate_data)
import numpy as np
import pandas as pd
# turn off auto wrapping of pandas dataframes
pd.set_option('display.expand_frame_repr', False)


def test_against_reference_implementation():
    num_levels = 6
    num_bufs = 4  # must be even
    xdim = 256
    ydim = 512
    stack_size = 100
    # img_stack = np.zeros((stack_size, xdim, ydim), dtype=int)
    img_stack = np.random.randint(1, 10, ((stack_size, xdim, ydim)))
    rois = np.zeros_like(img_stack[0])
    # make sure that the ROIs can be any integers greater than 1. They do not
    # have to start at 1 and be continuous
    rois[0:xdim//10, 0:ydim//10] = 5
    rois[xdim//10:xdim//5, ydim//10:ydim//5] = 3
    # run the correlation with the reference implementation
    g2, lag_steps = multi_tau_auto_corr(num_levels, num_bufs, rois, img_stack)
    # run the correlation with the accumulator version
    mt = MultiTauCorrelation(num_levels, num_bufs, rois)
    for img in img_stack:
        mt.process(img)

    # compare the results
    assert np.all(mt.g2 == g2)
    assert np.all(mt.lag_steps == lag_steps)

    # reset the partial data correlator and check the results again
    mt.reset()
    for img in img_stack:
        mt.process(img)

    # compare the results
    assert np.all(mt.g2 == g2)
    assert np.all(mt.lag_steps == lag_steps)
