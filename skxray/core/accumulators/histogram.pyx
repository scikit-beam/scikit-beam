"""
Histogram

General purpose histogram classes.
"""
cimport cython
import numpy as np
cimport numpy as np
import math
from libc.math cimport floor


ctypedef fused hnumtype:
    np.int_t
    np.float_t


# Histogramming classes
class histaxis:
    def __init__(self, nbin, low, high):
        self.low = low
        self.high = high
        self.nbin = nbin
        self.binsize = (high-low)/float(nbin)

    def bin(self, val):
        vala = np.asarray(val).reshape(-1)
        fidx = (vala - self.low) / self.binsize
        iidx = np.floor(fidx).astype(int)
        return iidx

    def values(self):
        return np.linspace(self.low+0.5*self.binsize, self.high-0.5*self.binsize, self.nbin)

class hist1d:

    def __init__(self, nbinx, xlow, xhigh):
        self.data = np.zeros(nbinx)
        #cdef float self.data[nbinx]
        self.nbinx = nbinx
        self.xaxis = histaxis(nbinx, xlow, xhigh)


    def fill(self, np.ndarray[hnumtype, ndim=1] xval, np.ndarray[hnumtype, ndim=1] weight):
        cdef np.ndarray[np.float_t, ndim=1] data = self.data
        cdef float low = self.xaxis.low
        cdef float high = self.xaxis.high
        cdef float binsize = self.xaxis.binsize
        cdef int i
        cdef int j
        cdef int xlen = len(xval)
        cdef np.float_t* pdata = <np.float_t*> data.data
        cdef hnumtype* px = <hnumtype*> xval.data
        cdef hnumtype* pw = <hnumtype*> weight.data
        for i in range(xlen):
            fillonecy(px[i], pw[i], pdata, low, high, binsize)
        return


cdef void fillonecy(hnumtype xval, hnumtype weight,
        np.float_t* pdata,
        float low, float high, float binsize):
    if not (low <= xval < high):
        return
    cdef int iidx
    iidx = int((xval - low) / binsize)
    pdata[iidx] += weight
    return
