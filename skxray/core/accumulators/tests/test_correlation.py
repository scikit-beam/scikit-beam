from skxray.core.correlation.correlation import multi_tau_auto_corr
from skxray.core.accumulators.correlation import (MultiTauCorrelation,
                                                  intermediate_data)
from skxray.core.accumulators.corr_gen import lazy_correlation
import numpy as np

# turn off auto wrapping of pandas dataframes
pd.set_option('display.expand_frame_repr', False)

num_levels = None
num_bufs = None
xdim = None
ydim = None
stack_size = None
img_stack = None
rois = None


def test_against_reference_implementation():
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


def setup():
    global num_levels, num_bufs, xdim, ydim, stack_size, img_stack, rois
    num_levels = 6
    num_bufs = 4  # must be even
    xdim = 256
    ydim = 512
    stack_size = 100
    img_stack = np.random.randint(1, 10, (stack_size, xdim, ydim))
    rois = np.zeros_like(img_stack[0])
    # make sure that the ROIs can be any integers greater than 1. They do not
    # have to start at 1 and be continuous
    rois[0:xdim//10, 0:ydim//10] = 5
    rois[xdim//10:xdim//5, ydim//10:ydim//5] = 3


def test_generator_against_reference():
    # run the correlation with the reference implementation
    g2, lag_steps = multi_tau_auto_corr(num_levels, num_bufs, rois, img_stack)
    gen = lazy_correlation(num_levels, num_bufs, img_stack, rois)
    all_res = list(gen)
    final_res = all_res[-1]
    assert np.all(final_res.g2 == g2)
    assert np.all(final_res.lag_steps == lag_steps)


