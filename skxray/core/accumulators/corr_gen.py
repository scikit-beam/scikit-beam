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
from skxray.core.correlation import correlation as corr
from collections import namedtuple
import numpy as np


results = namedtuple(
        'correlation_results',
        ['g2', 'lag_steps', 'internal_state']
)


class InternalCorrelationState:
    __slots__ = [
        'buf',
        'G',
        'past_intensity',
        'future_intensity',
        'img_per_level',
        'label_mask',
        # 'num_levels',
        # 'num_bufs',
        # 'num_pixels',
        # 'level',
        # 'buf_no',
        'track_level',
        'cur',
        'pixel_list',
        'label_mapping',
        'processed',
        # 'processing',
        # 'prev',
        # 'g2',
        # 'g_max',
        '__repr__',
    ]

    def __init__(self, num_levels, num_bufs, labels):
        self.label_mask, self.pixel_list = extract_label_indices(labels)
        # map the indices onto a sequential list of integers starting at 1
        self.label_mapping = {label: n for n, label in enumerate(
                np.unique(self.label_mask))}
        # remap the label mask to go from 0 -> max(_labels)
        for label, n in self.label_mapping.items():
            self.label_mask[self.label_mask == label] = n

        # G holds the un normalized auto- correlation result. We
        # accumulate computations into G as the algorithm proceeds.
        self.G = np.zeros(((num_levels + 1) * num_bufs / 2,
                           len(self.label_mapping)),
                          dtype=np.float64)
        # matrix for normalizing G into g2
        self.past_intensity = np.zeros_like(self.G)
        # matrix for normalizing G into g2
        self.future_intensity = np.zeros_like(self.G)
        # Ring buffer, a buffer with periodic boundary conditions.
        # Images must be keep for up to maximum delay in buf.
        self.buf = np.zeros((num_levels, num_bufs, len(self.pixel_list)),
                             dtype=np.float64)
        # to track how many images processed in each level
        self.img_per_level = np.zeros(num_levels, dtype=np.int64)
        # to track which levels have already been processed
        self.track_level = np.zeros(num_levels, dtype=bool)
        # to increment buffer
        self.cur = np.ones(num_levels, dtype=np.int64)
        # whether or not to process higher levels in multi-tau
        self.processed = 0


def _process(_state, num_bufs, num_pixels, level, buf_no):
    corr._process(_state.buf, _state.G, _state.past_intensity,
                  _state.future_intensity, _state.label_mask,
                  num_bufs, num_pixels, _state.img_per_level,
                  level, buf_no)


def lazy_correlation(image_iterable, num_levels, num_bufs, labels,
                     _state=None):
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
        _state = InternalCorrelationState(num_levels, num_bufs, labels)

    # create a shorthand reference to the results and state named tuple
    s = _state
    # stash the number of pixels in the mask
    num_pixels = np.bincount(s.label_mask)
    # Convert from num_levels, num_bufs to lag frames.
    tot_channels, lag_steps = multi_tau_lags(num_levels, num_bufs)

    # iterate over the images to compute multi-tau correlation
    for image in image_iterable:
        # Compute the correlations for all higher levels.
        level = 0

        # increment buffer
        s.cur[0] = (1 + s.cur[0]) % num_bufs

        # Put the ROI pixels into the ring buffer.
        s.buf[0, s.cur[0] - 1] = np.ravel(image)[s.pixel_list]
        buf_no = s.cur[0] - 1
        # Compute the correlations between the first level
        # (undownsampled) frames. This modifies G,
        # past_intensity, future_intensity,
        # and img_per_level in place!
        _process(s, num_bufs, num_pixels, level, buf_no)

        # check whether the number of levels is one, otherwise
        # continue processing the next level
        processing = num_levels > 1

        level = 1
        while processing:
            if not s.track_level[level]:
                s.track_level[level] = True
                processing = False
            else:
                prev = (1 + (s.cur[level - 1] - 2) % num_bufs)
                s.cur[level] = (
                    1 + s.cur[level] % num_bufs)

                # TODO clean this up. it is hard to understand
                s.buf[level, s.cur[level] - 1] = ((
                        s.buf[level - 1, prev - 1] +
                        s.buf[level - 1, s.cur[level - 1] - 1]
                    ) / 2
                )

                # make the track_level zero once that level is processed
                s.track_level[level] = False

                # call processing_func for each multi-tau level greater
                # than one. This is modifying things in place. See comment
                # on previous call above.
                buf_no = s.cur[level] - 1
                _process(s, num_bufs, num_pixels, level, buf_no)
                level += 1

                # Checking whether there is next level for processing
                processing = level < num_levels

        # If any past intensities are zero, then g2 cannot be normalized at
        # those levels. This if/else code block is basically preventing
        # divide-by-zero errors.
        if len(np.where(s.past_intensity == 0)[0]) != 0:
            g_max = np.where(s.past_intensity == 0)[0][0]
        else:
            g_max = s.past_intensity.shape[0]

        # Normalize g2 by the product of past_intensity and future_intensity
        g2 = (s.G[:g_max] /
              (s.past_intensity[:g_max] *
               s.future_intensity[:g_max]))
        
        s.processed += 1
        yield results(g2, lag_steps[:g_max], s)