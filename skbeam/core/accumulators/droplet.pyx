# ######################################################################
# Copyright (c) 2016, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Original code from Larry Lurio and Mark Sutton (see below for more   #
#            information, references to scientific papers)             #
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
''' This is a typical case of the porting of C code into python. Here
        we simply write wrapper functions. The key piece is the "extern from"
        command which allows one to make placeholder declarations as well as
        tell cython to find the necessary routines in the file described.

    This code has been ported from C code into python.
    Previous contributors (in chronological order, first is creator):
        -Larry Lurio (creator), July 2005, Northern Illinois University
        -Mark Sutton, June 2006, McGill University, Montreal, QC

        Literature using the droplet algorithm for beamlines:

        DeCaro, Curt, et al. "X-ray speckle visibility spectroscopy in the
        single-photon limit." Journal of synchrotron radiation 20.2 (2013):
        332-338.

        Livet, F., et al. "Using direct illumination CCDs as high-resolution
        area detectors for X-ray scattering." Nuclear Instruments and Methods
        in Physics Research Section A: Accelerators, Spectrometers, Detectors
        and Associated Equipment 451.3 (2000): 596-609.

        Hruszkewycz, S. O., et al. "High contrast x-ray speckle from
        atomic-scale order in liquids and glasses." Physical review letters
        109.18 (2012): 185502.

        Chushkin, Y., C. Caronna, and A. Madsen. "A novel event correlation
        scheme for X-ray photon correlation spectroscopy." Journal of Applied
        Crystallography 45.4 (2012): 807-813.
'''
import numpy as np
cimport numpy as np

cdef extern from "../../../src/droplet.c":
    void raw_expand(long int *img_out, long int *img_in, int ncol, int nrow);
    void raw_dropletfind(np.int_t * img_out, np.int_t *img_in, int ncol, int nrow, np.int_t *npeak);
    void raw_dropletanal(np.int_t *img, np.int_t* dimg, np.int_t *npix, \
            np.float_t *xcen, np.float_t *ycen, np.int_t * adus, np.int_t * idlist,\
            int npeak,int ncol, int nrow);

def dropletfind(np.ndarray[np.int_t, ndim=2, mode="c"] img):
    '''
    dropletfind(img)

    This routine takes a bitmap image (0 and 1's) and returns an image
    indentifying the droplets (connected regions).
    
    Each pixel in d is the droplet id of the cluster to which it belongs.
    nopeaks is assigned to the number of droplets in the image. The zeros of
    the bitmap may be considered as one big droplet (not necessarily connected)
    with id 0 (it is not counted in nopeaks). The id of each droplet is the
    index of the first pixel in the droplet as thats how the algorithm works.

    Parameters
    ----------
    img : the image. *Must be a numpy array of integers*

    Returns
    ------
    nopeaks : the number of peaks found

    img_out: An image of the same dimensions as input image with a unique
        nonzero integer id for each droplet found.  Surrounding regions are labeled
        with zero.

    See also
    --------
    dropletanal

    Notes
    -----
    The input image must be a 2D integer array of 0's and 1's

    Examples
    --------
    >>> img = np.random.random((10,10))
    >>> bimg = (img < .3).astype(int)
    >>> npeaks, dimg = dropletfind(bimg)

    Debugging (Remove me)
    ---------
    For debugging, possible problems (remove me in final version):
        1. The array ordering might not always be numpy's typical ordering
            (fastest varying index last). Perhaps check for this.
        2. Check that assumed data types are correct.
    '''
    if img.dtype != np.int_:
        raise TypeError("Error, wrong data type, must be int. Got: {}".format(img.dtype))

    d = img.shape
    cdef np.ndarray[np.int_t, ndim=1, mode="c"] npeak = np.array([0]);
    cdef np.ndarray[np.int_t, ndim=2, mode="c"] img_out = np.zeros_like(img);

    # faster varying index is second in numpy, so flip
    raw_dropletfind(&img_out[0,0], &img[0,0], d[1], d[0], &npeak[0]);

    return npeak[0], img_out


def dropletanal(np.ndarray[np.int_t, ndim=2, mode="c"] img, \
                np.ndarray[np.int_t, ndim=2, mode="c"] dimg,\
                int npeaks):
    '''
    dropletanal(img, dropletmap, npeaks)

    This routine analyzes a droplet map to collect statistics about the
    droplets suitable for photon analysis. Given the image and its droplet map,
    it returns arrays of values for each droplet.

    Parameters
    ----------
    img : The data you want to collect statistics from (must be floating point array)

    dimg : The image with droplet id's assigned per pixel (as returned by dropletfind)

    npeaks : The number of droplets (as returned by dropletfind)


    Returns
    -------
    npix : Number of pixels in each droplet

    xcen : intensity (in img) weighted center of mass for x 

    ycen : intensity (in img) weighted center of mass for y

    adus : the summed intensity in each droplet

    idlist : the unique id of each droplet. nopeaks is the number of droplets
        in the image. As a side effect it modifies the dropletmap into cycles so
        further processing will become easier. The cycles link pixels of same
        droplet. So the code:
     	pos=dropletmap(id);
	do { //do stuff with pixel img(pos)
	   pos=dropletmap(pos)}
	   }
	while(pos!=id);
        will loop through pixels of droplet id. Save a copy of dropletmap
        if you want to use it later. (Really is only easier to process this
        way in C code.)

    See also
    --------
    dropletfind

    Notes
    -----
    img and dimg must be a 2D array of integers.

    Examples
    --------
    >>> img = (np.random.random((10,10))*100).astype(int)
    >>> bimg = (img < 10).astype(int)
    >>> npeaks, dimg = dropletfind(bimg)
    >>> npix, xcen, ycen, adus, idlist = dropletanal(img, dimg, npeaks)

    Debugging (remove me for final version)
    ---------
    Could perhaps allow img to be a 2D array of floats?
 '''
    if img.dtype != np.int_:
        raise TypeError("Error, wrong data type, must be int. Got: {}".format(img.dtype))

    if dimg.dtype != np.int_:
        raise TypeError("Error, wrong data type, must be int. Got: {}".format(dimg.dtype))

    cdef np.ndarray[np.int_t, ndim=1, mode="c"] npix = np.zeros(npeaks, dtype=np.int_)
    cdef np.ndarray[np.float_t, ndim=1, mode="c"] xcen = np.zeros(npeaks, dtype=np.float_)
    cdef np.ndarray[np.float_t, ndim=1, mode="c"] ycen = np.zeros(npeaks, dtype=np.float_)
    cdef np.ndarray[np.int_t, ndim=1, mode="c"] adus = np.zeros(npeaks, dtype=np.int_)
    cdef np.ndarray[np.int_t, ndim=1, mode="c"] idlist = np.zeros(npeaks, dtype=np.int_)

    d = img.shape

    raw_dropletanal(&img[0,0], &dimg[0,0], &npix[0], &xcen[0], &ycen[0],\
                    &adus[0], &idlist[0], npeaks, d[1], d[0])

    return npix, xcen, ycen, adus, idlist

#TODO : Make a version that allows a floating point image
    # adus would also have to be float
