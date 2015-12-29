from __future__ import division
import numpy as np
cimport cython
cimport numpy as np
cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b


@cython.boundscheck(False)
cdef _process(np.ndarray[double, ndim=3] buf,
              np.ndarray[double, ndim=2] G,
              np.ndarray[double, ndim=2] past_intensity_norm,
              np.ndarray[double, ndim=2] future_intensity_norm,
              np.ndarray[long, ndim=1] label_mask,
              long num_bufs,
              np.ndarray[long, ndim=1] num_pixels,
              np.ndarray[long, ndim=1] img_per_level,
              long level,
              long buf_no):
    """Optimized cython implementation of correlation._process

    This helper function calculates G, past_intensity_norm and
    future_intensity_norm at each level, symmetric normalization is used.

    .. warning :: This modifies inputs in place.

    Parameters
    ----------
    buf : array
        image data array to use for correlation
    G : array
        matrix of auto-correlation function without normalizations
    past_intensity_norm : array
        matrix of past intensity normalizations
    future_intensity_norm : array
        matrix of future intensity normalizations
    label_mask : array
        labeled array where all nonzero values are ROIs
    num_bufs : int, even
        number of buffers(channels)
    num_pixels : array
        number of pixels in certain roi's
        roi's, dimensions are : [number of roi's]X1
    img_per_level : array
        to track how many images processed in each level
    level : int
        the current multi-tau level
    buf_no : int
        the current buffer number
    """
    img_per_level[level] += 1
    # in multi-tau correlation, the subsequent levels have half as many
    # buffers as the first
    cdef int i_min = num_bufs // 2 if level else 0
    cdef int t_index
    cdef int delay_no
    cdef int i
    cdef np.ndarray past_img, future_img, corr
    # cdef np.float_t [:,:,:] data = buf

    for i in range(i_min, int_min(img_per_level[level], num_bufs)):
        # compute the index into the autocorrelation matrix
        t_index = level * num_bufs // 2 + i

        delay_no = (buf_no - i) % num_bufs
        # get the images for correlating
        past_img = buf[level][delay_no]
        future_img = buf[level][buf_no]
        corr = past_img * future_img
        for w, arr in zip([corr, past_img, future_img],
                          [G, past_intensity_norm, future_intensity_norm]):
            binned = np.bincount(label_mask, weights=w)
            arr[t_index] += ((binned / num_pixels - arr[t_index]) /
                             (img_per_level[level] - i))

    return None  # modifies arguments in place!


def process_wrapper(buf, G, past_intensity_norm, future_intensity_norm,
                    label_mask, num_bufs, num_pixels, img_per_level, level,
                    buf_no):
    _process(buf, G, past_intensity_norm, future_intensity_norm,
             label_mask, num_bufs, num_pixels, img_per_level, level,
             buf_no)


