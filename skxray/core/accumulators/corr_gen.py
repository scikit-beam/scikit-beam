# # Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
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
from __future__ import absolute_import, division, print_function
from skxray.core import roi, utils
from skxray.core.correlation.correlation import _process
import numpy as np
import time as ttime


def multi_tau_auto_corr_partial_data(num_levels, num_bufs, labels, images,
                                     previous=None):
    """
    This function computes one-time correlations.

    It uses a scheme to achieve long-time correlations inexpensively
    by downsampling the data, iteratively combining successive frames.

    The longest lag time computed is num_levels * num_bufs.

    Parameters
    ----------
    num_levels : int
        how many generations of downsampling to perform, i.e.,
        the depth of the binomial tree of averaged frames

    num_bufs : int, must be even
        maximum lag step to compute in each generation of
        downsampling

    labels : array
        labeled array of the same shape as the image stack;
        each ROI is represented by a distinct label (i.e., integer)

    images : iterable of 2D arrays
        dimensions are: (rr, cc)

    previous : tuple
        holds last result (g2, lag_steps) of correlation,
        and current state (G, past_intensity_norm, future_intensity_norm, buf)

    Returns
    -------
    g2 : array
        matrix of normalized intensity-intensity autocorrelation
        shape (num_levels, number of labels(ROI))

    lag_steps : array
        delay or lag steps for the multiple tau analysis
        shape num_levels

    Notes
    -----

    The normalized intensity-intensity time-autocorrelation function
    is defined as

    :math ::
        g_2(q, t') = \frac{<I(q, t)I(q, t + t')> }{<I(q, t)>^2}

    ; t' > 0

    Here, I(q, t) refers to the scattering strength at the momentum
    transfer vector q in reciprocal space at time t, and the brackets
    <...> refer to averages over time t. The quantity t' denotes the
    delay time

    This implementation is based on code in the language Yorick
    by Mark Sutton, based on published work. [1]_

    References
    ----------

    .. [1] D. Lumma, L. B. Lurio, S. G. J. Mochrie and M. Sutton,
        "Area detector based photon correlation in the regime of
        short data batches: Data reduction for dynamic x-ray
        scattering," Rev. Sci. Instrum., vol 70, p 3274-3289, 2000.

    """
    # In order to calculate correlations for `num_bufs`, images must be
    # kept for up to the maximum lag step. These are stored in the array
    # buffer. This algorithm only keeps number of buffers and delays but
    # several levels of delays number of levels are kept in buf. Each
    # level has twice the delay times of the next lower one. To save
    # needless copying, of cyclic storage of images in buf is used.

    if num_bufs % 2 != 0:
        raise ValueError("number of channels(number of buffers) in "
                         "multiple-taus (must be even)")

    if hasattr(images, 'frame_shape'):
        # Give a user-friendly error if we can detect the shape from pims.
        if labels.shape != images.frame_shape:
            raise ValueError("Shape of the image stack should be equal to"
                             " shape of the labels array")

    # get the pixels in each label
    label_mask, pixel_list = roi.extract_label_indices(labels)

    num_rois = np.max(label_mask)

    # number of pixels per ROI
    num_pixels = np.bincount(label_mask, minlength=(num_rois + 1))
    num_pixels = num_pixels[1:]

    if np.any(num_pixels == 0):
        raise ValueError("Number of pixels of the required roi's"
                         " cannot be zero, "
                         "num_pixels = {0}".format(num_pixels))

    if previous is None:
        # G holds the un normalized auto-correlation result. We
        # accumulate computations into G as the algorithm proceeds.
        G = np.zeros(((num_levels + 1) * num_bufs / 2, num_rois),
                     dtype=np.float64)

        # matrix of past intensity normalizations
        past_intensity_norm = np.zeros(
                ((num_levels + 1) * num_bufs / 2, num_rois),
                dtype=np.float64)

        # matrix of future intensity normalizations
        future_intensity_norm = np.zeros(
                ((num_levels + 1) * num_bufs / 2, num_rois),
                dtype=np.float64)

        # Ring buffer, a buffer with periodic boundary conditions.
        # Images must be keep for up to maximum delay in buf.
        buf = np.zeros((num_levels, num_bufs, np.sum(num_pixels)),
                       dtype=np.float64)

        # to track processing each level
        track_level = np.zeros(num_levels)

        # to increment buffer
        cur = np.ones(num_levels, dtype=np.int64)

        # to track how many images processed in each level
        img_per_level = np.zeros(num_levels, dtype=np.int64)

        start_time = ttime.time()  # used to log the computation time (
        # optionally)

    else:
        (G, past_intensity_norm, future_intensity_norm, buf,
         cur, img_per_level, track_level) = previous[2:]

    for n, img in enumerate(images):

        cur[0] = (1 + cur[0]) % num_bufs  # increment buffer

        # Put the image into the ring buffer.
        buf[0, cur[0] - 1] = (np.ravel(img))[pixel_list]

        # Compute the correlations between the first level
        # (undownsampled) frames. This modifies G,
        # past_intensity_norm, future_intensity_norm,
        # and img_per_level in place!
        _process(buf, G, past_intensity_norm,
                 future_intensity_norm, label_mask,
                 num_bufs, num_pixels, img_per_level,
                 level=0, buf_no=cur[0] - 1)

        # check whether the number of levels is one, otherwise
        # continue processing the next level
        processing = num_levels > 1

        # Compute the correlations for all higher levels.
        level = 1
        while processing:
            if not track_level[level]:
                track_level[level] = 1
                processing = False
            else:
                prev = 1 + (cur[level - 1] - 2) % num_bufs
                cur[level] = 1 + cur[level] % num_bufs

                buf[level, cur[level] - 1] = (buf[level - 1, prev - 1] +
                                              buf[level - 1,
                                                  cur[level - 1] - 1]) / 2

                # make the track_level zero once that level is processed
                track_level[level] = 0

                # call the _process function for each multi-tau level
                # for multi-tau levels greater than one
                # Again, this is modifying things in place. See comment
                # on previous call above.
                _process(buf, G, past_intensity_norm,
                         future_intensity_norm, label_mask,
                         num_bufs, num_pixels, img_per_level,
                         level=level, buf_no=cur[level] - 1, )
                level += 1

                # Checking whether there is next level for processing
                processing = level < num_levels

    # the normalization factor
    if len(np.where(past_intensity_norm == 0)[0]) != 0:
        g_max = np.where(past_intensity_norm == 0)[0][0]
    else:
        g_max = past_intensity_norm.shape[0]

    # g2 is normalized G
    g2 = (G[:g_max] / (past_intensity_norm[:g_max] *
                       future_intensity_norm[:g_max]))

    # Convert from num_levels, num_bufs to lag frames.
    tot_channels, lag_steps = utils.multi_tau_lags(num_levels, num_bufs)
    lag_steps = lag_steps[:g_max]

    previous = (
        g2, lag_steps, G, past_intensity_norm, future_intensity_norm, buf,
        cur, img_per_level, track_level)
    yield previous
