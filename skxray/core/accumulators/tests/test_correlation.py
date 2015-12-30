from skxray.core.correlation.correlation import (multi_tau_auto_corr,
                                                 intermediate_data)
from skxray.core.accumulators.correlation import MultiTauCorrelation
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
    gen = multi_tau_auto_corr(num_levels, num_bufs, rois, img_stack)
    res = list(gen)
    g2, lag_steps = res[-1]
    # run the correlation with the accumulator version
    mt = MultiTauCorrelation(num_levels, num_bufs, rois)
    res2 = []
    for img in img_stack:
        mt.process(img)
        res2.append(mt.get_current_state())

    equal = []
    for accum, full in zip(res2, res):
        for accum_item, full_item in zip(accum, full):
            equal.append(np.all(accum_item == full_item))

    equal = [np.all(accum_item == full_item)
             for accum, full in zip(res2, res)
             for accum_item, full_item in zip(accum, full)]
    df = pd.DataFrame(np.asarray(equal).reshape(len(res2), len(res2[0])),
                      columns=intermediate_data._fields)

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

    raise
