# ######################################################################
# multi_tau_auto_corr Original code(in Yorick):                        #
# @author: Mark Sutton                                                 #
#                                                                      #
# two_time_corr Original code                                          #
# @author: Yugang Zhang                                                #
#                                                                      #
# Developed at the NSLS-II, Brookhaven National Laboratory             #
# Developed by Sameera K. Abeykoon, February 2014                      #
#                                                                      #
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

"""

This module is for functions specific to time correlation

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np
import numpy.ma as ma
import logging
import time

import skxray.core as core
logger = logging.getLogger(__name__)


def multi_tau_auto_corr(num_levels, num_bufs, labels, images):
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

    Returns
    -------
    g2 : array
        matrix of one-time correlation
        shape (num_levels, number of labels(ROI))

    lag_steps : array
        delay or lag steps for the multiple tau analysis
        shape num_levels

    Note
    ----

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

    if labels.shape != images.operands[0].shape[1:]:
        raise ValueError("Shape of the image stack should be equal to"
                         " shape of the labels array")

    # get the pixels in each label
    label_mask, pixel_list = extract_label_indices(labels)

    num_rois = np.max(label_mask)

    # number of pixels per ROI
    num_pixels = np.bincount(label_mask, minlength=(num_rois+1))
    num_pixels = num_pixels[1:]

    if np.any(num_pixels == 0):
        raise ValueError("Number of pixels of the required roi's"
                         " cannot be zero, "
                         "num_pixels = {0}".format(num_pixels))

    # G holds the un normalized auto-correlation result. We
    # accumulate computations into G as the algorithm proceeds.
    G = np.zeros(((num_levels + 1)*num_bufs/2, num_rois),
                 dtype=np.float64)

    # matrix of past intensity normalizations
    past_intensity_norm = np.zeros(((num_levels + 1)*num_bufs/2, num_rois),
                                   dtype=np.float64)

    # matrix of future intensity normalizations
    future_intensity_norm = np.zeros(((num_levels + 1)*num_bufs/2, num_rois),
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

    start_time = time.time()  # used to log the computation time (optionally)

    for n, img in enumerate(images.operands[0]):

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
                                                  cur[level - 1] - 1])/2

                # make the track_level zero once that level is processed
                track_level[level] = 0

                # call the _process function for each multi-tau level
                # for multi-tau levels greater than one
                # Again, this is modifying things in place. See comment
                # on previous call above.
                _process(buf, G, past_intensity_norm,
                         future_intensity_norm, label_mask,
                         num_bufs, num_pixels, img_per_level,
                         level=level, buf_no=cur[level]-1,)
                level += 1

                # Checking whether there is next level for processing
                processing = level < num_levels

    # ending time for the process
    end_time = time.time()

    logger.info("Processing time for {0} images took {1} seconds."
                "".format(n, (end_time - start_time)))

    # the normalization factor
    if len(np.where(past_intensity_norm == 0)[0]) != 0:
        g_max = np.where(past_intensity_norm == 0)[0][0]
    else:
        g_max = past_intensity_norm.shape[0]

    # g2 is normalized G
    g2 = (G[:g_max] / (past_intensity_norm[:g_max] *
                       future_intensity_norm[:g_max]))

    # Convert from num_levels, num_bufs to lag frames.
    tot_channels, lag_steps = core.multi_tau_lags(num_levels, num_bufs)
    lag_steps = lag_steps[:g_max]

    return g2, lag_steps


def _process(buf, G, past_intensity_norm, future_intensity_norm,
             label_mask, num_bufs, num_pixels, img_per_level, level, buf_no):
    """
    Internal helper function. This modifies inputs in place.

    This helper function calculates G, past_intensity_norm and
    future_intensity_norm at each level, symmetric normalization is used.

    Parameters
    ----------
    buf : array
        image data array to use for correlation

    G : array
        matrix of auto-correlation function without
        normalizations

    past_intensity_norm : array
        matrix of past intensity normalizations

    future_intensity_norm : array
        matrix of future intensity normalizations

    label_mask : array
        labels of the required region of interests(roi's)

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

    Notes
    -----
    :math ::
        G   = <I(t)I(t + delay)>

    :math ::
        past_intensity_norm = <I(t)>

    :math ::
        future_intensity_norm = <I(t + delay)>

    """
    img_per_level[level] += 1

    # in multi-tau correlation other than first level all other levels
    #  have to do the half of the correlation
    if level == 0:
        i_min = 0
    else:
        i_min = num_bufs//2

    for i in range(i_min, min(img_per_level[level], num_bufs)):
        t_index = level*num_bufs/2 + i

        delay_no = (buf_no - i) % num_bufs

        past_img = buf[level, delay_no]
        future_img = buf[level, buf_no]

        #  get the matrix of auto-correlation function without normalizations
        tmp_binned = (np.bincount(label_mask,
                                  weights=past_img*future_img)[1:])
        G[t_index] += ((tmp_binned / num_pixels - G[t_index]) /
                       (img_per_level[level] - i))

        # get the matrix of past intensity normalizations
        pi_binned = (np.bincount(label_mask,
                                 weights=past_img)[1:])
        past_intensity_norm[t_index] += ((pi_binned/num_pixels
                                         - past_intensity_norm[t_index]) /
                                         (img_per_level[level] - i))

        # get the matrix of future intensity normalizations
        fi_binned = (np.bincount(label_mask,
                                 weights=future_img)[1:])
        future_intensity_norm[t_index] += ((fi_binned/num_pixels
                                           - future_intensity_norm[t_index]) /
                                           (img_per_level[level] - i))

    return None  # modifies arguments in place!


def extract_label_indices(labels):
    """
    This will find the label's required region of interests (roi's),
    number of roi's count the number of pixels in each roi's and pixels
    list for the required roi's.

    Parameters
    ----------
    labels : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).

    Returns
    -------
    label_mask : array
        1D array labeling each foreground pixel
        e.g., [1, 1, 1, 1, 2, 2, 1, 1]

    indices : array
        1D array of indices into the raveled image for all
        foreground pixels (labeled nonzero)
        e.g., [5, 6, 7, 8, 14, 15, 21, 22]
    """
    img_dim = labels.shape

    # TODO Make this tighter.
    w = np.where(np.ravel(labels) > 0)
    grid = np.indices((img_dim[0], img_dim[1]))
    pixel_list = np.ravel((grid[0] * img_dim[1] + grid[1]))[w]

    # discard the zeros
    label_mask = labels[labels > 0]

    return label_mask, pixel_list


def two_time_corr():

def _process_two_time(lev,bufno,n):
        num[lev]+=1
        if lev==0:
            imin = 0
        else:
            imin = num_bufs/2
        for i in range(imin, min(num[lev], num_bufs) ):
            ptr = lev * num_bufs/2+i
            delayno = (bufno-i)%num_bufs #//cyclic buffers
            IP=buf[lev,delayno]
            IF=buf[lev,bufno]
            I_t12 = (np.histogram(qind, bins=num_rois, weights= IF*IP))[0]
            I_t1  =  (np.histogram(qind, bins=num_rois, weights= IP))[0]
            I_t2  =  (np.histogram(qind, bins=num_rois, weights= IF))[0]
            tind1 = (n-1);tind2=(n -dly[ptr] -1)

            if not isinstance(n, int ):
                nshift = 2**(lev-1)
                for i in range( -nshift+1, nshift +1 ):
                    #print tind1+i
                    g12[ int(tind1 + i), int(tind2 + i) ] =I_t12/( I_t1 * I_t2) * nopr
            else:
                #print tind1
                g12[ tind1, tind2 ]  =   I_t12/( I_t1 * I_t2) * nopr

    def insertimg_twotime(self, n, norm=None, print_=False):
        cur[0]=1+cur[0]%num_bufs  # increment buffer
        fp = FILENAME + '%04d'%n

        if img_format=='EDF':fp+='.edf';img= get_edf( fp ) -  DK
        elif img_format=='TIFF':fp+='.tiff';img = scipy.misc.imread(fp,flatten=1)
        else:img= cpopen( n= n,prefix= 'data_', inDir=DATA_DIR)

        if print_:print 'The insert image %s is %s' %(n,fp)
        buf[0, cur[0]-1 ]=img.flatten()[pixellist]
        img=[] #//save space
        countl[0] = 1+ countl[0]
        current_img_time = n - begframe +1
        self.process_two_time(lev=0, bufno=cur[0]-1,n=current_img_time )
        time_ind[0].append(  current_img_time   )
        processing=1
        lev=1
        while processing:
            if cts[lev]:
                prev=  1+ (cur[lev-1]-1-1+num_bufs)%num_bufs
                cur[lev]=  1+ cur[lev]%num_bufs
                countl[lev] = 1+ countl[lev]
                buf[lev,cur[lev]-1] = ( buf[lev-1,prev-1] + buf[lev-1,cur[lev-1]-1] ) /2
                cts[lev]=0
                t1_idx=   (countl[lev]-1) *2
                current_img_time = ((time_ind[lev-1])[t1_idx ] +  (time_ind[lev-1])[t1_idx +1 ] )/2.
                time_ind[lev].append(  current_img_time      )
                self.process_two_time(lev= lev, bufno= cur[lev]-1,n=current_img_time )
                lev+=1
                #//Since this level finished, test if there is a next level for processing
                if lev<num_levels:processing = 1
                else:processing = 0
            else:
                cts[lev]=1      #// set flag to process next time
                processing=0    #// can stop until more images are accumulated


    def two_time_corr(num_levels, num_bufs, num_rois):
        global buf, num, cts, cur, g12, countl
        global Ndel, Npix
        global time_ind  #generate a time-frame for each level
        global g12x, g12y, g12z #for interpolate
        start_time = time.time()
        buf = np.zeros([num_levels, num_bufs, num_pixels])  #// matrix of buffers, for store img
        cts = np.zeros(num_levels)
        cur = np.ones(num_levels) * num_bufs
        countl = np.array(np.zeros(  num_levels ),dtype='int')
        g12 =  np.zeros([noframes, noframes, num_rois] )
        g12x = []
        g12y = []
        g12z = []
        num= np.array(np.zeros(num_levels),dtype='int')
        time_ind ={key: [] for key in range(num_levels)}
        ttx=0
        for n in range(1, noframes +1 ):   ##do the work here
            insertimg_twotime(begframe + n - 1, print_=print_)
            if  n %(noframes/10) ==0:
                sys.stdout.write("#")
                sys.stdout.flush()
        for q in range(num_rois):
            x0 =  g12[:,:,q]
            g12[:,:,q] = tril(x0) +  tril(x0).T - diag(diag(x0))
        elapsed_time = time.time() - start_time
        return g12, (elapsed_time/60.)


