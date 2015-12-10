"""
Histogram

General purpose histogram classes.
"""
cimport cython
import numpy as np
cimport numpy as np
import math
from libc.math cimport floor
from ..utils import bin_edges_to_centers

import logging
logger = logging.getLogger(__name__)


ctypedef fused xnumtype:
    np.int_t
    np.float_t

ctypedef fused ynumtype:
    np.int_t
    np.float_t

ctypedef fused wnumtype:
    np.int_t
    np.float_t


class Histogram:
    def __init__(self, binlowhigh, *args):
        """

        Parameters
        ----------
        binlowhigh : iterable
            nbin, low, high = binlowhigh
            nbin is the number of bins
            low is the left most edge
            high is the right most edge
        args : iterable
            Extra instances of binlowhigh that correspond to extra dimensions
            in the Histogram

        Notes
        -----
        The right most bin is half open
        """
        if len(args) > 1:
            raise NotImplementedError(
                "This class does not yet support higher dimensional histograms "
                "than 2D"
            )
        logger.debug('binlowhigh = {}'.format(binlowhigh))
        logger.debug('args = {}'.format(args))
        nbins = []
        lows = []
        highs = []
        for bin, low, high in [binlowhigh] + list(args):
            nbins.append(bin)
            lows.append(low)
            highs.append(high)

        logger.debug("nbins = {}".format(nbins))

        # create the numpy array to hold the results
        self._values = np.zeros(nbins, dtype=float)
        self.ndims = len(nbins)
        binsizes = [(high - low) / nbin for high, low, nbin
                    in zip(highs, lows, nbins)]
        logger.debug("nbins = {}".format(nbins))
        # store everything in a numpy array
        self._nbins = np.array(nbins, dtype=np.dtype('i')).reshape(-1)
        self._lows = np.array(lows, dtype=np.dtype('f')).reshape(-1)
        self._highs = np.array(highs, dtype=np.dtype('f')).reshape(-1)
        self._binsizes = np.array(binsizes, dtype=np.dtype('f')).reshape(-1)


    def reset(self):
        """Fill the histogram array with 0
        """
        self._values.fill(0)

    def fill(self, *coords, weights=1):
        """

        Parameters
        ----------
        coords : iterable of numpy arrays
            The length of coords is equivalent to the dimensionality of
            the histogram.
        weights

        Returns
        -------

        """
        # check our arguments
        if len(coords) != self.ndims:
            emsg = "Incorrect number of arguments.  Received {} expected {}."
            raise ValueError(emsg.format(len(coords), self.ndims))

        weights = np.asarray(weights).reshape(-1)

        nexpected = len(coords[0])
        for x in coords:
            if len(x) != nexpected:
                emsg = "Coordinate arrays must have the same length."
                raise ValueError(emsg)
        if len(weights) != 1 and len(weights) != nexpected:
            emsg = "Weights must be scalar or have the same length as coordinates."
            raise ValueError(emsg)

        if len(coords) == 1:
            # compute a 1D histogram
            self._fill1d(coords[0], weights)
        elif len(coords) == 2:
            # compute a 2D histogram!
            self._fill2d(coords[0], coords[1], weights)
        else:
            # do the generalized ND histogram
            raise NotImplementedError()
        return

    def _fill1d(self, np.ndarray[xnumtype, ndim=1] xval,
                np.ndarray[wnumtype, ndim=1] weight):
        cdef np.ndarray[np.float_t, ndim=1] data = self.values
        cdef float low = self._lows[0]
        cdef float high = self._highs[0]
        cdef float binsize = self._binsizes[0]
        cdef int i
        cdef int xlen = len(xval)
        cdef np.float_t* pdata = <np.float_t*> data.data
        cdef xnumtype* px = <xnumtype*> xval.data
        cdef wnumtype* pw = <wnumtype*> weight.data
        if weight.size == 1:
            for i in range(xlen):
                fillonecy(px[i], pw[0], pdata, low, high, binsize)
        else:
            for i in range(xlen):
                fillonecy(px[i], pw[i], pdata, low, high, binsize)
        return

    def _fill2d(self, np.ndarray[xnumtype, ndim=1] xval,
                np.ndarray[ynumtype, ndim=1] yval,
                np.ndarray[wnumtype, ndim=1] weight):
        # cdef np.ndarray[np.float_t, ndim=2] data = self.values
        cdef double [:,:] data = self.values
        cdef float [:] low = self._lows
        cdef float [:] high = self._highs
        cdef float [:] binsize = self._binsizes
        cdef int i
        cdef int xlen = len(xval)
        cdef int ylen = len(yval)
        # cdef float [:,:] pdata =
        # cdef np.float_t* pdata = <np.float_t*> data.data
        cdef xnumtype* px = <xnumtype*> xval.data
        cdef ynumtype* py = <ynumtype*> yval.data
        cdef wnumtype* pw = <wnumtype*> weight.data
        if weight.size == 1:
            for i in range(xlen):
                xidx = find_indices(px[i], low[0], high[0], binsize[0])
                if xidx == -1:
                    continue
                yidx = find_indices(py[i], low[1], high[1], binsize[1])
                if yidx == -1:
                    continue
                data[xidx][yidx] += pw[0]
                # pdata[yidx * xlen + xidx] += pw[0]
        else:
            for i in range(xlen):
                xidx = find_indices(px[i], low[0], high[0], binsize[0])
                if xidx == -1:
                    continue
                yidx = find_indices(py[i], low[1], high[1], binsize[1])
                if yidx == -1:
                    continue
                data[xidx][yidx] += pw[i]
                # pdata[yidx * xlen + xidx] += pw[i]
        return

    @property
    def values(self):
        return self._values

    @property
    def edges(self):
        return [np.linspace(low, high, nbin+1) for nbin, low, high
                in zip(self._nbins, self._lows, self._highs)]

    @property
    def centers(self):
        return [bin_edges_to_centers(edge) for edge in self.edges]


cdef long find_indices(xnumtype pos, float low, float high, float binsize):
    if not (low <= pos < high):
        return -1
    return int((pos - low) / binsize)

cdef void fillonecy(xnumtype xval, wnumtype weight,
                    np.float_t* pdata,
                    float low, float high, float binsize):
    iidx = find_indices(xval, low, high, binsize)
    if iidx == -1:
        return
    pdata[iidx] += weight
    return

#TODO function interface
#TODO generator interface
#TODO docs!
#TODO implement ND histogram
#TODO examples
