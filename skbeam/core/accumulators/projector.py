from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np


class Projector(object):
    """
    Class to project an arbitray ndarray onto an axis using a
    user-specified bin for each element of the array.
    """

    def __init__(self, bin_values, nbins, norm, weights=None):
        """
        Parameters
        ----------
        bin_values : np.ndarray
            For each pixel, this is the value that determines which bin
            a pixel's intensity will contribute to.  It can be an arbitrary
            array of floats/ints which will be discretized into nbins equally
            sized bins.
        nbins : int
            The number of bins to project into.
        norm : bool
            If set to true, normalize projection by the sum of the weights
            of all pixels contributing to a bin.  If weights=None, then
            divide by the number of pixels contributing to a bin.
        weights : np.ndarray, optional
            A weight for each pixel with the same shape as bin_values.
            These can be set to zero to ignore a pixel, similar to a mask.
            If not provided, each pixel is assigned a weight of 1.
        """

        self._bin_values = bin_values
        self._weights = weights
        self._nbins = nbins
        self._norm = norm

        if weights is None:
            self._bv = self._bin_values
        else:
            if not (weights.shape == bin_values.shape):
                raise ValueError('"weights" and "bin_values" must have the '
                                 'same shape', weights.shape,
                                 self._bin_values.shape)
            # _ibv = included bin values
            # this is used to select only unmasked pixels in the analysis
            # (i.e. those with weight!=0)
            self._ibv = (weights != 0)
            self._bv = self._bin_values[self._ibv]

        binvalRange = self._bv.max() - self._bv.min()
        # add a little bit to not have the self._bv.max() value create
        # an extra bin in the np.floor statement below
        binvalRange *= 1.0+1e-9
        self.bin_width = binvalRange / self._nbins

        self._bin_assignments = np.floor((self._bv - self._bv.min()) /
                                         self.bin_width)
        self._bin_assignments = (self._bin_assignments.astype(np.int32).
                                 flatten())
        if self._norm:
            if weights is None:
                self._normalization_array = (
                    (np.bincount(self._bin_assignments))
                    .astype(np.float))
            else:
                wt = self._weights[self._ibv]
                self._normalization_array = (
                    (np.bincount(self._bin_assignments,
                                 weights=wt.flatten()))
                    .astype(np.float))

            # to avoid divide-by-zero
            self._normalization_array[self._normalization_array == 0] = np.nan

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
            The average intensity in the bin.
        """

        if not (image.shape == self._bin_values.shape):
            raise ValueError('"image" and "bin_values" must have the '
                             'same shape', image.shape, self._bin_values.shape)

        if self._weights is None:
            histogram = np.bincount(self._bin_assignments,
                                    weights=image.flatten())
        else:
            weights = image * self._weights
            histogram = np.bincount(self._bin_assignments,
                                    weights=weights[self._ibv].flatten())

        if self._norm:
            histogram /= self._normalization_array

        return histogram

    @property
    def centers(self):
        """
        Returns bin centers of projected histogram.

        Returns:
        --------
        bin_centers : np.ndarray, float
            The center of each bin.
        """
        return (np.arange(self._nbins) + 0.5) * self.bin_width + (
            self._bv.min())


class RadialProjector(Projector):
    """
    Class to project a 2D image onto a radial axis.
    """

    def __init__(self, xsize, ysize, nbins, norm, xc=None, yc=None, rmin=None,
                 rmax=None, phimin=None, phimax=None, weights=None,
                 cartesian=True):
        """
        Parameters:
        -----------
        xsize,ysize: int
            shape of image in pixels.
        nbins: int
            number of bins in projected histogram.
        norm : bool
            If set to true, normalize projection by the sum of the weights
            of all pixels contributing to a bin.  If weights=None, then
            divide by the number of pixels contributing to a bin.
        xc,yc: int
            location (in pixels) of origin (default: image center).
        rmin,rmax: int
            radial range to include in projection, in pixels
            (default: no limits).
        phimin,phimax: float
            phi range to include in projection in the range
            (-180,180) degrees (default: no limits).  The location
            of phi=0 depends on the setting of the "cartesian"
            parameter.
        weights: np.ndarray
            weight to be applied to each pixel in
            image.  this can be used as a mask if the weight is
            set to zero. must be the same shape as the image being projected.
        cartesian: bool
            if True, use "cartesian" ordering: with x corresponding
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

        super(RadialProjector, self).__init__(rpix, nbins, norm, weights)
