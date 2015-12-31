from ..roi import extract_label_indices
import numpy as np
from collections import namedtuple
from ..correlation.correlation import _process as pyprocess
from ..utils import multi_tau_lags


intermediate_data = namedtuple(
    'intermediate_data',
    ['image_num', 'max_images', 'G', 'buf', 'past_intensity_norm',
     'future_intensity_norm', 'label_mask', 'num_bufs', 'num_pixels',
     'img_per_level', 'level', 'buf_no', 'prev', 'cur', 'track_level', 'g2',
     'lag_steps'])


class MultiTauCorrelation:

    def __init__(self, num_levels, num_bufs, labels, processing_func=pyprocess):
        """
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
        """
        self._g2 = None
        self._level = 0
        self._processed = -1
        self._processing = False
        self._prev = None
        self._processing_func = pyprocess
        self._new_levels_bufs_or_labels(num_levels, num_bufs, labels)

    @property
    def processing_func(self):
        return self._processing_func

    @processing_func.setter
    def processing_func(self, proc):
        self._processing_func = proc

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels
        # get the pixels in each label
        self._label_mask, self._pixel_list = extract_label_indices(self._labels)
        # map the indices onto a sequential list of integers starting at 1
        self._label_mapping = {label: n for n, label in enumerate(
                np.unique(self._label_mask))}
        # remap the label mask to go from 0 -> max(self._labels)
        for label, n in self._label_mapping.items():
            self._label_mask[self._label_mask == label] = n
        self._num_pixels = np.bincount(self._label_mask)

    def _new_levels_bufs_or_labels(self, num_levels=None, num_bufs=None,
                                   labels=None):
        if num_levels is not None:
            self._num_levels = num_levels
        if num_bufs is not None:
            self._num_bufs = num_bufs
        if labels is not None:
            self.labels = labels

        if num_levels or num_bufs:
            # Convert from num_levels, num_bufs to lag frames.
            self._tot_channels, self._lag_steps = multi_tau_lags(
                    self._num_levels, self._num_bufs)

        # G holds the un normalized auto- correlation result. We
        # accumulate computations into G as the algorithm proceeds.
        self._G = np.zeros(((self._num_levels + 1) * self._num_bufs / 2,
                           len(self._label_mapping)), dtype=np.float64)

        # matrix of past intensity normalizations
        self._past_intensity = np.zeros_like(self._G)

        # matrix of future intensity normalizations
        self._future_intensity = np.zeros_like(self._G)

        # Ring buffer, a buffer with periodic boundary conditions.
        # Images must be keep for up to maximum delay in buf.
        self._buf = np.zeros((self._num_levels, self._num_bufs,
                              len(self._pixel_list)),
                              dtype=np.float64)

        # to track processing each level
        self._track_level = np.zeros(self._num_levels, dtype=bool)

        # to increment buffer
        self._cur = np.ones(self._num_levels, dtype=np.int64)

        # to track how many images processed in each level
        self._img_per_level = np.zeros(self._num_levels, dtype=np.int64)

    @property
    def g2(self):
        return self._g2

    @property
    def lag_steps(self):
        return self._lag_steps[:self._g_max]

    def reset(self):
        """Clear the internal state"""
        # zero out all the arrays
        self._G[:] = 0
        self._past_intensity[:] = 0
        self._future_intensity[:] = 0
        self._buf[:] = 0
        self._track_level[:] = 0
        self._cur[:] = 0
        self._img_per_level[:] = 0
        # reset all scalar values
        self._buf_no = 0
        self._level = 0
        self._processed = -1
        # reset all boolean values
        self._processing = False

    @property
    def intermediate_data(self):
        return intermediate_data(
            self._processed, -1, self._G, self._buf, self._past_intensity,
            self._future_intensity, self._label_mask, self._num_bufs,
            self._num_pixels, self._img_per_level, self._level, self._buf_no,
            self._prev, self._cur, self._track_level, self.g2, self.lag_steps
        )

    def process(self, img):
        # Compute the correlations for all higher levels.
        self._level = 0

        # increment buffer
        self._cur[0] = (1 + self._cur[0]) % self._num_bufs

        # Put the ROI pixels into the ring buffer.
        self._buf[0, self._cur[0] - 1] = np.ravel(img)[self._pixel_list]
        self._buf_no = self._cur[0] - 1
        # Compute the correlations between the first level
        # (undownsampled) frames. This modifies G,
        # past_intensity_norm, future_intensity_norm,
        # and img_per_level in place!
        self._processing_func(
            self._buf, self._G, self._past_intensity,
            self._future_intensity, self._label_mask,
            self._num_bufs, self._num_pixels, self._img_per_level,
            self._level, buf_no=self._buf_no)

        # check whether the number of levels is one, otherwise
        # continue processing the next level
        self._processing = self._num_levels > 1

        self._level = 1
        self._prev = None
        while self._processing:
            if not self._track_level[self._level]:
                self._track_level[self._level] = True
                self._processing = False
            else:
                self._prev = (1 + (self._cur[self._level - 1] - 2) %
                              self._num_bufs)
                self._cur[self._level] = (
                    1 + self._cur[self._level] % self._num_bufs)

                self._buf[self._level, self._cur[self._level] - 1] = ((
                        self._buf[self._level-1, self._prev-1] +
                        self._buf[self._level-1, self._cur[self._level-1]-1]
                    ) / 2
                )

                # make the track_level zero once that level is processed
                self._track_level[self._level] = False

                # call processing_func for each multi-tau level greater
                # than one. This is modifying things in place. See comment
                # on previous call above.
                self._buf_no = self._cur[self._level] - 1
                self._processing_func(
                        self._buf, self._G, self._past_intensity,
                        self._future_intensity, self._label_mask,
                        self._num_bufs, self._num_pixels,
                        self._img_per_level, self._level,
                        buf_no=self._buf_no)
                self._level += 1

                # Checking whether there is next level for processing
                self._processing = self._level < self._num_levels

        # If any past intensities are zero, then g2 cannot be normalized at
        # those levels. This if/else code block is basically preventing
        # divide-by-zero errors.
        if len(np.where(self._past_intensity == 0)[0]) != 0:
            self._g_max = np.where(self._past_intensity == 0)[0][0]
        else:
            self._g_max = self._past_intensity.shape[0]

        # Normalize g2 by the product of past_intensity and future_intensity
        self._g2 = (self._G[:self._g_max] /
                    (self._past_intensity[:self._g_max] *
                     self._future_intensity[:self._g_max]))
        self._processed += 1
