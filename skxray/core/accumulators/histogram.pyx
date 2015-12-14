"""
Histogram

General purpose histogram classes.
"""
cimport cython
import numpy as np
cimport numpy as np
from ..utils import bin_edges_to_centers

import logging
logger = logging.getLogger(__name__)

DEF MAX_DIMENSIONS = 10

ctypedef fused xnumtype:
    np.int_t
    np.float_t

ctypedef fused ynumtype:
    np.int_t
    np.float_t

ctypedef fused wnumtype:
    np.int_t
    np.float_t


cdef void* _getarrayptr(np.ndarray a):
    return <void*> a.data


class Histogram:

    _always_use_fillnd = False      # FIXME remove this

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
        if 1 + len(args) > MAX_DIMENSIONS:
            emsg = "Cannot create histogram of more than {} dimensions."
            raise ValueError(emsg.format(MAX_DIMENSIONS))
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
            emsg = ("Weights must be scalar or have the same length "
                    "as coordinates.")
            raise ValueError(emsg)
        if self._always_use_fillnd:
            self._fillnd(coords, weights)
            return
        if len(coords) == 1:
            # compute a 1D histogram
            self._fill1d(coords[0], weights)
        elif len(coords) == 2:
            # compute a 2D histogram!
            self._fill2d(coords[0], coords[1], weights)
        else:
            # do the generalized ND histogram
            self._fillnd(coords, weights)
        return


    def _fill1d(self, np.ndarray[xnumtype, ndim=1] xval,
                np.ndarray[wnumtype, ndim=1] weight):
        cdef np.ndarray[np.float_t, ndim=1] data = self.values
        cdef float low = self._lows[0]
        cdef float high = self._highs[0]
        cdef float binsize = self._binsizes[0]
        cdef int i
        cdef int wstride = 0 if weight.size == 1 else 1
        cdef int xlen = len(xval)
        for i in range(xlen):
            xidx = find_indices(xval[i], low, high, binsize)
            if xidx == -1:
                continue
            data[xidx] += weight[wstride * i]
        return


    def _fill2d(self, np.ndarray[xnumtype, ndim=1] xval,
                np.ndarray[ynumtype, ndim=1] yval,
                np.ndarray[wnumtype, ndim=1] weight):
        cdef double [:,:] data = self.values
        cdef float [:] low = self._lows
        cdef float [:] high = self._highs
        cdef float [:] binsize = self._binsizes
        cdef int i
        cdef int xlen = len(xval)
        cdef int ylen = len(yval)
        cdef int wstride = 0 if weight.size == 1 else 1
        for i in range(xlen):
            xidx = find_indices(xval[i], low[0], high[0], binsize[0])
            if xidx == -1:
                continue
            yidx = find_indices(yval[i], low[1], high[1], binsize[1])
            if yidx == -1:
                continue
            data[xidx][yidx] += weight[wstride * i]
        return


    def _fillnd(self, coords, np.ndarray[wnumtype, ndim=1] weight):
        # allocate pointer arrays per each supported numerical types
        cdef np.int_t* aint_ptr[MAX_DIMENSIONS]
        cdef int aint_stride[MAX_DIMENSIONS]
        cdef int aint_count = 0
        cdef np.float_t* afloat_ptr[MAX_DIMENSIONS + 1]
        cdef int afloat_stride[MAX_DIMENSIONS]
        cdef int afloat_count = 0
        # determine order of coordinate arrays according to their
        # numerical data type
        numtypes = [np.dtype(int), np.dtype(float)]
        numtypeindex = {tp : i for i, tp in enumerate(numtypes)}
        ctypeindices = [numtypeindex.get(x.dtype, 999) for x in coords]
        coordsorder = np.argsort(ctypeindices, kind='mergesort')
        istrides = np.asarray(self._values.strides, dtype=np.int32)[coordsorder]
        istrides //= self._values.itemsize
        cdef int [:] dataindexstrides = istrides
        mylows = self._lows[coordsorder]
        myhighs = self._highs[coordsorder]
        mybinsizes = self._binsizes[coordsorder]
        cdef float [:] low = mylows
        cdef float [:] high = myhighs
        cdef float [:] binsize = mybinsizes
        cdef np.float_t* data = <np.float_t*> _getarrayptr(self._values)
        # distribute coordinates in each dimension according to their
        # numerical type.  follow the same order as in numtypes.
        for x, dstride in zip(coords, dataindexstrides):
            if x.dtype == np.int:
                aint_ptr[aint_count] = <np.int_t*> _getarrayptr(x)
                aint_stride[aint_count] = dstride
                aint_count += 1
            elif x.dtype == np.float:
                afloat_ptr[afloat_count] = <np.float_t*> _getarrayptr(x)
                afloat_stride[afloat_count] = dstride
                afloat_count += 1
            else:
                emsg = "Numpy arrays of type {} are not supported."
                raise TypeError(emsg.format(x.dtype))
        cdef int i, j, k
        cdef int wstride = 0 if weight.size == 1 else 1
        cdef int xlen = len(coords[0])
        cdef int xidx, widx, didx
        for i in range(xlen):
            didx = 0
            for k in range(aint_count):
                j = k
                xidx = find_indices(aint_ptr[k][i],
                                    low[j], high[j], binsize[j])
                if xidx == -1:
                    didx = -1
                    break
                didx += dataindexstrides[j] * xidx
            if didx == -1:
                continue
            for k in range(afloat_count):
                j = k + aint_count
                xidx = find_indices(afloat_ptr[k][i],
                                    low[j], high[j], binsize[j])
                if xidx == -1:
                    didx = -1
                    break
                didx += dataindexstrides[j] * xidx
            if didx == -1:
                continue
            widx = wstride * i
            data[didx] += weight[widx]
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
#TODO examples
#TODO Can we support ND histogram for mixed coordinate types?
