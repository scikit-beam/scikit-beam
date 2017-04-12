from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np


class BinImage(object):
    """
    Class to bin an ndarray onto an axis using a
    user-specified bin for each element of the array.
    """

    def __init__(self, bin_values, norm, nbins, vmin=None, vmax=None,
                 weights=None):
        """
        Parameters
        ----------
        bin_values : np.ndarray
            For each pixel, this is the value that determines which bin
            a pixel's intensity will contribute to.  It can be an arbitrary
            array of floats/ints which will be discretized into nbins equally
            sized bins.
        norm : bool
            If set to true, normalize histogram by the sum of the weights
            of all pixels contributing to a bin (see "weights" parameter).
        nbins : int
            The number of histogram bins.
        vmin/vmax : int/float, optional
            Lower/Upper limits of returned histogram.  If either is not
            specified one will be computed from bin_values, ignoring
            pixels where weight==0.
        weights : np.ndarray, optional
            A weight for each pixel with the same shape as bin_values.
            These can be set to zero to ignore a pixel, similar to a mask.
            If not provided, each pixel is assigned a weight of 1.
        """

        self._bin_values = bin_values
        self._weights = weights
        self._nbins = nbins
        self._norm = None
        self._vmin = vmin
        self._vmax = vmax

        # modify weights to mask additional pixels flagged by vmin/vmax limits
        if self._weights is None and (self._vmin is not None or
                                      self._vmax is not None):
            self._weights = np.ones_like(bin_values)
        if self._vmin is not None:
            self._weights[bin_values < self._vmin] = 0
        if self._vmax is not None:
            # exclude vmax values, consistent with histogram pattern
            self._weights[bin_values >= self._vmax] = 0

        # compute the bin_values, perhaps using the weights
        if self._weights is None:
            self._bv = self._bin_values
        else:
            if not (self._weights.shape == bin_values.shape):
                raise ValueError('"weights" and "bin_values" must have the '
                                 'same shape', self._weights.shape,
                                 self._bin_values.shape)
            # _ibv = included bin values
            # this is used to select only unmasked pixels in the analysis
            # (i.e. those with weight!=0)
            self._ibv = (self._weights != 0)
            self._bv = self._bin_values[self._ibv]

        # compute the vmin/vmax values, if necessary
        if self._vmin is None:
            self._vmin = self._bv.min()
        if self._vmax is None:
            # user didn't supply vmax value so do it ourselves.
            # add a little bit to not have the vmax value create
            # an extra bin in the np.floor statement below
            self._vmax = self._bv.max()
            self._vmax += (self._vmax-self._vmin) * 1e-9

        self.bin_width = (self._vmax - self._vmin) / float(self._nbins)
        self._bin_assignments = np.floor((self._bv - self._vmin) /
                                         self.bin_width)
        self._bin_assignments = (self._bin_assignments.astype(np.int32).
                                 reshape(-1))
        if norm:
            if self._weights is None:
                self._norm = (
                    (np.bincount(self._bin_assignments))
                    .astype(np.float))
            else:
                wt = self._weights[self._ibv]
                self._norm = (
                    (np.bincount(self._bin_assignments,
                                 weights=wt.reshape(-1)))
                    .astype(np.float))

            # to avoid divide-by-zero
            self._norm[self._norm == 0] = np.nan

        return

    def __call__(self, image):
        """
        Bin pixel intensities.

        Parameters
        ----------
        image : np.ndarray
            The intensity at each pixel.
        Returns
        -------
        histogram : np.ndarray
            The intensity in the bin.
        """

        if not (image.shape == self._bin_values.shape):
            raise ValueError('"image" and "bin_values" must have the '
                             'same shape', image.shape, self._bin_values.shape)

        if self._weights is None:
            histogram = np.bincount(self._bin_assignments,
                                    weights=image.reshape(-1))
        else:
            weights = image * self._weights
            histogram = np.bincount(self._bin_assignments,
                                    weights=weights[self._ibv].reshape(-1))

        if self._norm is not None:
            histogram /= self._norm

        return histogram

    @property
    def centers(self):
        """
        Returns bin centers of histogram.

        Returns:
        --------
        bin_centers : np.ndarray, float
            The center of each bin.
        """
        return (np.arange(self._nbins) + 0.5) * self.bin_width + (
            self._vmin)


class RadialBinImage(BinImage):
    """
    Create a 1-dimensional histogram by binning a 2-dimensional
    image in radius.
    """

    def __init__(self, xsize, ysize, nbins, norm, xc=None, yc=None, rmin=None,
                 rmax=None, phimin=None, phimax=None, weights=None,
                 cartesian=True):
        """
        Parameters:
        -----------
        xsize,ysize: int
            shape of image in pixels.  see "cartesian" parameter
            for definition of x/y.
        nbins: int
            number of bins in histogram.
        norm : bool
            If set to true, normalize histogram by the sum of the weights
            of all pixels contributing to a bin.  If weights=None, then
            divide by the number of pixels contributing to a bin.
        xc,yc: int, optional
            location (in pixels) of origin (default: image center).
            see "cartesian" parameter for definition of x/y.
        rmin,rmax: int, optional
            radial range to include in histogram, in pixels
            (default: will be computed from pixels with weights!=0).
        phimin,phimax: float, optional
            phi range to include.  Values are in the range
            (-180,180) degrees (default: no limits).  Phi is
            computed as arctan(y/x) where the meaning of y and x
            is determined by the setting of the "cartesian" parameter.
        weights: 2-dimensional np.ndarray, optional
            weight to be applied to each pixel when binning.
            this can be used as a mask if the weight is set to zero.
            must be the same shape as the image being projected.
            (default: 1 for all pixels)
        cartesian: bool, optional
            if True, use "cartesian" ordering, with x corresponding
            to matrix columns and y corresponding to matrix rows.
            Otherwise the opposite ("matrix" ordering). (default: True)
        """

        if weights is not None:
            if not (len(weights.shape) == 2):
                raise ValueError('"weights" must be a 2-dimensional array')

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

        if weights is None and (None not in (phimin, phimax)):
            weights = np.ones((ysize, xsize))  # "cartesian"
        if phimin is not None:
            weights[phipix < phimin] = 0
        if phimax is not None:
            # adopt the usual histogramming convention of excluding
            # the upper edge
            weights[phipix >= phimax] = 0

        super(RadialBinImage, self).__init__(rpix, norm, nbins, rmin, rmax,
                                             weights)
