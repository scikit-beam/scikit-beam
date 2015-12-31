# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Redistribution and use in source and binary forms, with or without   #
# modification, are permitted provided that the following conditions   #
# are met:                                                             #
#                                                                      #
# * Redistributions of source code must retain the above copyright     #
#   notice, this list of conditions and the following disclaimer.      #
#                                                                      #
# * Redistributions in binary form must reproduce the above copyright  #
#   notice this list of conditions and the following disclaimer in     #
#   the documentation and/or other materials provided with the         #
#   distribution.                                                      #
#                                                                      #
# * Neither the name of the Brookhaven Science Associates, Brookhaven  #
#   National Laboratory nor the names of its contributors may be used  #
#   to endorse or promote products derived from this software without  #
#   specific prior written permission.                                 #
#                                                                      #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS  #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT    #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS    #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE       #
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,           #
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR   #
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)   #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,  #
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OTHERWISE) ARISING   #
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE   #
# POSSIBILITY OF SUCH DAMAGE.                                          #
########################################################################
from __future__ import absolute_import, division, print_function

from skxray.core.utils import multi_tau_lags
from skxray.core.roi import extract_label_indices
from collections import namedtuple
import numpy as np

correlation_state = namedtuple(
        'correlation_state',
        ['buf', 'G', 'past_intensity', 'future_intensity',
         'label_mask', 'num_bufs', 'num_pixels', 'img_per_level', 'level',
         'buf_no', 'track_level', 'cur', 'pixel_list'])


class InternalCorrelationState:
    pass


results = namedtuple(
        'correlation_results',
        ['g2', 'lag_steps', 'internal_state']
)


def _process(_state: correlation_state):
    corr._process(_state.buf, _state.G, _state.past_intensity_norm,
                  _state.future_intensity_norm, _state.label_mask,
                  _state.num_bufs, _state.num_pixels, _state.img_per_level,
                  _state.level, _state.buf_no)


def _init_correlation_state(num_levels, num_bufs, labels):
    # get the pixels in each label
    label_mask, pixel_list = extract_label_indices(labels)
    # map the indices onto a sequential list of integers starting at 1
    label_mapping = {label: n for n, label in enumerate(
            np.unique(label_mask))}
    # remap the label mask to go from 0 -> max(_labels)
    for label, n in label_mapping.items():
        label_mask[label_mask == label] = n
    num_pixels = np.bincount(label_mask)

    # G holds the un normalized auto- correlation result. We
    # accumulate computations into G as the algorithm proceeds.
    G = np.zeros(((num_levels + 1) * num_bufs / 2, len(label_mapping)),
                 dtype=np.float64)

    return correlation_state(
        G=G,
        # Ring buffer, a buffer with periodic boundary conditions.
        # Images must be keep for up to maximum delay in buf.
        buf=np.zeros((num_levels, num_bufs, len(pixel_list)),
                     dtype=np.float64),
        # matrix for normalizing G into g2
        past_intensity=np.zeros_like(G),
        # matrix for normalizing G into g2
        future_intensity=np.zeros_like(G),
        # the 1-D list that keeps track of which pixels in `pixel_list`
        # correspond to which ROI
        label_mask=label_mask,
        # the 1-D list that indexes into the raveled image to get all pixels
        # in ROIs
        pixel_list=pixel_list,
        num_bufs=num_bufs,
        num_pixels=num_pixels,
        # to track how many images processed in each level
        img_per_level=np.zeros(num_levels, dtype=np.int64),
        # the current level being computed
        level=0,
        # the current position in the ring buffer
        buf_no=0,
        # to track which levels have already been processed
        track_level=np.zeros(num_levels, dtype=bool),
        # to increment buffer
        cur=np.ones(num_levels, dtype=np.int64)
    )



def lazy_correlation(image_iterable, num_levels, num_bufs, labels,
                     _state: correlation_state = None):
    """Generator implementation of 1-time multi-tau correlation

    Parameters
    ----------
    num_levels : int
        how many generations of downsampling to perform, i.e., the depth of
        the binomial tree of averaged frames
    num_bufs : int, must be even
        maximum lag step to compute in each generation of downsampling
    labels : array
        Labeled array of the same shape as the image stack.
        Each ROI is represented by sequential integers starting at one.  For
        example, if you have four ROIs, they must be labeled 1, 2, 3,
        4. Background is labeled as 0
    labels : array
        Labeled array of the same shape as the image stack.
        Each ROI is represented by sequential integers starting at one.  For
        example, if you have four ROIs, they must be labeled 1, 2, 3,
        4. Background is labeled as 0
    images : iterable of 2D arrays
    _state : namedtuple, optional
        _state is a bucket for all of the internal state of the generator.
        It is part of the `results` object that is yielded from this
        generator

    Yields
    ------
    state : namedtuple
        A 'results' object that contains:
        - the normalized correlation, `g2`
        - the times at which the correlation was computed, `lag_steps`
        - and all of the internal state, `final_state`, which is a
          `correlation_state` namedtuple
    """
    if _state is None:
        _state = _init_correlation_state(num_levels, num_bufs, labels)

    # create a shorthand reference to the results and state named tuple
    s = _state
    # Convert from num_levels, num_bufs to lag frames.
    tot_channels, lag_steps = multi_tau_lags(num_levels, num_bufs)
    # create the results namedtuple
    r = results(np.zeros_like(s.past_intensity), lag_steps, _state)

    # iterate over the images to compute multi-tau correlation
    for image in image_iterable:
        # Compute the correlations for all higher levels.
        s.level = 0

        # increment buffer
        s.cur[0] = (1 + s.cur[0]) % s.num_bufs

        # Put the ROI pixels into the ring buffer.
        s.buf[0, s.cur[0] - 1] = np.ravel(image)[s.pixel_list]
        s.buf_no = s.cur[0] - 1
        # Compute the correlations between the first level
        # (undownsampled) frames. This modifies G,
        # past_intensity_norm, future_intensity_norm,
        # and img_per_level in place!
        s.processing_func(
            s.buf, s.G, s.past_intensity,
            s.future_intensity, s.label_mask,
            s.num_bufs, s.num_pixels, s.img_per_level,
            s.level, s.buf_no)

        # check whether the number of levels is one, otherwise
        # continue processing the next level
        s.processing = s.num_levels > 1

        s.level = 1
        s.prev = None
        while s.processing:
            if not s.track_level[s.level]:
                s.track_level[s.level] = True
                s.processing = False
            else:
                s.prev = (1 + (s.cur[s.level - 1] - 2) %
                          s.num_bufs)
                s.cur[s.level] = (
                    1 + s.cur[s.level] % s.num_bufs)

                # TODO clean this up. it is hard to understand
                s.buf[s.level, s.cur[s.level] - 1] = ((
                        s.buf[s.level - 1, s.prev - 1] +
                        s.buf[s.level - 1, s.cur[s.level - 1] - 1]
                    ) / 2
                )

                # make the track_level zero once that level is processed
                s.track_level[s.level] = False

                # call processing_func for each multi-tau level greater
                # than one. This is modifying things in place. See comment
                # on previous call above.
                s.buf_no = s.cur[s.level] - 1
                s.processing_func(
                        s.buf, s.G, s.past_intensity,
                        s.future_intensity, s.label_mask,
                        s.num_bufs, s.num_pixels,
                        s.img_per_level, s.level,
                        s.buf_no)
                s.level += 1

                # Checking whether there is next level for processing
                s.processing = s.level < s.num_levels

        # If any past intensities are zero, then g2 cannot be normalized at
        # those levels. This if/else code block is basically preventing
        # divide-by-zero errors.
        if len(np.where(s.past_intensity == 0)[0]) != 0:
            s.g_max = np.where(s.past_intensity == 0)[0][0]
        else:
            s.g_max = s.past_intensity.shape[0]

        # Normalize g2 by the product of past_intensity and future_intensity
        r.g2 = (s.G[:s.g_max] /
                (s.past_intensity[:s.g_max] *
                 s.future_intensity[:s.g_max]))
        s.processed += 1
        yield r
