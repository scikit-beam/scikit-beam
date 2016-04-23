#!/usr/bin/env python

from __future__ import division
import numpy as np

"""
authors: Mikhail Dubrovin, TJ Lane, Christopher O'Grady
"""


class Projector(object):

    def __init__(self, bin_values, nbins, weights=None, norm=True):
        """
        Parameters
        ----------
        bin_values : np.ndarray
            For each pixel, this is the value of that determines which bin
            a pixel's intensity will contribute to
        weights : np.ndarray
            A weight for each pixel with the same shape as bin_values (set
            to zero to ignore a pixel)
        nbins : int
            The number of bins to project into
        """

        self.bin_values = bin_values
        self.weights = weights
        self.nbins = nbins
        self.norm = norm

        # _ibv = included bin values
        # this is used to select only unmasked pixels in the analysis
        # (i.e. those with weight!=0)
        if weights is not None:
            self._ibv = (weights != 0)
        else:
            self._ibv = np.s_[:]  # slice that returns arr unchanged

        bv = self.bin_values[self._ibv]
        wt = self.weights[self._ibv]

        binvalRange = bv.max() - bv.min()
        # add a little bit to not have the bv.max() value create an extra bin
        # in the np.floor statement below
        binvalRange *= 1.0+1e-9
        self.bin_width = binvalRange / (float(self.nbins))

        self._bin_assignments = np.floor((bv - bv.min()) /
                                         self.bin_width).astype(np.int32)
        smallval = np.finfo(np.float).eps
        if weights is None:
            self._normalization_array = \
                (np.bincount(self._bin_assignments.flatten()) + smallval) \
                .astype(np.float)
        else:
            self._normalization_array = \
                (np.bincount(self._bin_assignments.flatten(),
                             weights=wt.flatten()) + smallval) \
                 .astype(np.float)

        assert self.nbins == self._bin_assignments.max()+1, \
            'incorrect bin assignments (%d %d)' % (self.nbins,
                                                   self._bin_assignments.max())
        self._normalization_array = self._normalization_array[:self.nbins]

        return

    def __call__(self, image):
        """
        Bin pixel intensities.

        Parameters
        ----------
        image : np.ndarray
            The intensity at each pixel
        Returns
        -------
        histogram : np.ndarray
            The average intensity in the bin
        """

        if not (image.shape == self.bin_values.shape):
            raise ValueError('"image" and "bin_values" must have the '
                             'same shape', image.shape, self.bin_values.shape)

        if self.weights is None:
            weights = image.flatten()
        else:
            weights = image.flatten() * self.weights.flatten()
            if not (image.shape == self.weights.shape):
                raise ValueError('`image` and `weights` must have the '
                                 'same shape')

        histogram = np.bincount(self._bin_assignments.flatten(),
                                weights=weights[self._ibv.flatten()])

        if self.norm:
            histogram /= self._normalization_array

        assert histogram.shape[0] == self.nbins, '%d %d' % \
            (histogram.shape[0], self.nbins)

        return histogram

    @property
    def bin_centers(self):
        """
        Returns:
        --------
        bin_centers : ndarray, float
            The center of each bin.
        """
        return (np.arange(self.nbins) + 0.5) * self.bin_width + \
            self.bin_values[self._ibv].min()


class RadialProjector(Projector):
    """
    Project a 2D image onto a radial axis.
    """

    def __init__(self, xsize, ysize, nbins, xc=None, yc=None, rmin=None,
                 rmax=None, phimin=None, phimax=None, weights=None,
                 norm=True, cartesian=True):
        """
        Parameters:
        -----------
        xsize,ysize:   shape of image in pixels
        nbins:         number of bins in projected histogram
        xc,yc:         location (in pixels) of origin (default: image center)
        rmin,rmax:     radial range to include in projection, in pixels
                       (default: no limits)
        phimin,phimax: phi range to include in projection in the range
                       (-180,180) degrees (default: no limits)
        weights:       np.ndarray.  weight to be applied to each pixel in
                       image.  this can be used as a mask if the weight is
                       set to zero
        norm:          boolean indicating whether bin entries in the projected
                       histogram should be divided by weights (number of
                       pixels, in the case where the weights are 1)
        cartesian:     if True, use "cartesian" ordering: with x corresponding
                       to matrix columns and y corresponding to matrix rows.
                       Otherwise the opposite ("matrix" ordering).
        """

        if not cartesian:
            # switch from matrix to cartesian by swapping axes
            xc, yc = yc, xc
            xsize, ysize = ysize, xsize
        xc = xsize//2 if xc is None else xc
        yc = ysize//2 if yc is None else yc
        x = np.arange(xsize)-xc
        y = np.arange(ysize)-yc
        xgrid, ygrid = np.meshgrid(x, y)  # "cartesian"

        rpix = np.sqrt(xgrid**2 + ygrid**2)

        if phimin is not None or phimax is not None:
            if cartesian:
                phipix = np.arctan2(ygrid, xgrid) * 180 / np.pi
            else:
                phipix = np.arctan2(xgrid, ygrid) * 180 / np.pi

        if weights is None and (None not in (phimin, phimax, rmin, rmax)):
            weights = np.ones((ysize, xsize))  # "cartesian"
        if phimin is not None:
            weights[phipix < phimin] = 0
        if phimax is not None:
            weights[phipix > phimax] = 0
        if rmin is not None:
            weights[rpix < rmin] = 0
        if rmax is not None:
            weights[rpix > rmax] = 0

        super(RadialProjector, self).__init__(rpix, nbins, weights, norm=norm)
