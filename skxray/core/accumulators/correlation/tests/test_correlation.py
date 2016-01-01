from __future__ import absolute_import, print_function, division
from skxray.core.accumulators.correlation import lazy_multi_tau
from skxray.core.accumulators.correlation .pyprocess import pyprocess
from skxray.core.accumulators.correlation .cyprocess import cyprocess

import numpy as np

num_levels = None
num_bufs = None
xdim = None
ydim = None
stack_size = None
img_stack = None
rois = None


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


def test_lazy_multi_tau():
    for func in [pyprocess, cyprocess]:
        yield _lazy_multi_tau, func


def _lazy_multi_tau(processing_func):
    setup()
    # run the correlation on the full stack
    full_gen = lazy_multi_tau(
        num_levels, num_bufs, rois, img_stack, processing_func=processing_func)
    for full_result in full_gen:
        pass

    # make sure we have essentially zero correlation in the images,
    # since they are random integers
    assert np.average(full_result.g2-1) < 0.01

    # run the correlation on the first half
    gen_first_half = lazy_multi_tau(
        num_levels, num_bufs, rois, img_stack[:stack_size//2],
        processing_func=processing_func)
    for first_half_result in gen_first_half:
        pass
    # run the correlation on the second half by passing in the state from the
    # first half
    gen_second_half = lazy_multi_tau(
        num_levels, num_bufs, rois, img_stack[stack_size//2:],
        processing_func=processing_func,
        _state=first_half_result.internal_state
    )

    for second_half_result in gen_second_half:
        pass

    assert np.all(full_result.g2 == second_half_result.g2)
