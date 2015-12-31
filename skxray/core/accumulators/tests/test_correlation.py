from __future__ import absolute_import, print_function, division
from skxray.core.correlation.correlation import multi_tau_auto_corr
from skxray.core.accumulators.correlation import lazy_multi_tau
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


def test_generator_against_reference():
    # run the correlation with the reference implementation
    full_g2, full_lag_steps = multi_tau_auto_corr(num_levels, num_bufs, rois,
                                                  img_stack)
    full_gen = lazy_multi_tau(img_stack, num_levels, num_bufs, rois)
    full_res = list(full_gen)
    assert np.all(full_res[-1].g2 == full_g2)
    assert np.all(full_res[-1].lag_steps == full_lag_steps)

    # now let's do half the correlation and compare
    midpoint = stack_size//2
    # compute the correlation with the reference implementation on the first
    # half of the image stack
    first_half_g2, first_half_lag_steps = multi_tau_auto_corr(
            num_levels, num_bufs, rois, img_stack[:midpoint])
    # and compute it with the generator implementation on the first half
    first_half_gen = lazy_multi_tau(img_stack[:midpoint], num_levels, num_bufs, rois)
    first_half_res = list(first_half_gen)
    # compare the results
    assert np.all(first_half_res[-1].g2 == first_half_g2)
    assert np.all(first_half_res[-1].lag_steps == first_half_lag_steps)

    # now continue on the second half
    second_half_gen = lazy_multi_tau(
            img_stack[midpoint:], num_levels, num_bufs, rois,
            _state=first_half_res[-1].internal_state)
    second_half_res = list(second_half_gen)

    # and make sure the results are the same as running the generator on the
    # full stack of images
    assert np.all(second_half_res[-1].g2 == full_res[-1].g2)
    assert np.all(second_half_res[-1].lag_steps == full_res[-1].lag_steps)
