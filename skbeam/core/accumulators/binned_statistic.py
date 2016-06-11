from __future__ import division, print_function, absolute_import

import warnings

import numpy as np
from scipy._lib.six import callable
from collections import namedtuple


BinnedStatisticResult = namedtuple('BinnedStatisticResult',
                                   ('statistic', 'bin_edges', 'binnumber'))

BinnedStatisticddResult = namedtuple('BinnedStatisticddResult',
                                     ('statistic', 'bin_edges',
                                      'binnumber'))


class BinnedStatistic(object):
    def __init__(self, x, statistic='mean',
                 bins=10, range=None):
        """
        A refactored version of scipy.stats.binned_statistic to improve
        performance for the case where binning doesn't need to be
        re-initialized on every call.

        Compute a binned statistic for a set of data.

        This is a generalization of a histogram function.  A histogram divides
        the space into bins, and returns the count of the number of points in
        each bin.  This function allows the computation of the sum, mean,
        median, or other statistic of the values within each bin.

        Parameters
        ----------
        x : array_like
            A sequence of values to be binned.
        statistic : string or callable, optional
            The statistic to compute (default is 'mean').
            The following statistics are available:

              * 'mean' : compute the mean of values for points within each bin.
                Empty bins will be represented by NaN.
              * 'median' : compute the median of values for points within each
                bin. Empty bins will be represented by NaN.
              * 'count' : compute the count of points within each bin.  This is
                identical to an unweighted histogram.  `values` array is not
                referenced.
              * 'sum' : compute the sum of values for points within each bin.
                This is identical to a weighted histogram.
              * function : a user-defined function which takes a 1D array of
                values, and outputs a single numerical statistic. This function
                will be called on the values in each bin.  Empty bins will be
                represented by function([]), or NaN if this returns an error.
        bins : int or sequence of scalars, optional
            If `bins` is an int, it defines the number of equal-width bins in
            the given range (10 by default).  If `bins` is a sequence, it
            defines the bin edges, including the rightmost edge, allowing for
            non-uniform bin widths.  Values in `x` that are smaller than lowest
            bin edge are assigned to bin number 0, values beyond the highest
            bin are assigned to ``bins[-1]``.
        range : (float, float) or [(float, float)], optional
            The lower and upper range of the bins.  If not provided, range
            is simply ``(x.min(), x.max())``.  Values outside the range are
            ignored.

        See Also
        --------
        numpy.histogram, binned_statistic_2d, binned_statistic_dd

        Notes
        -----
        All but the last (righthand-most) bin is half-open.  In other words, if
        `bins` is ``[1, 2, 3, 4]``, then the first bin is ``[1, 2)`` (including
        1, but excluding 2) and the second ``[2, 3)``.  The last bin, however,
        is ``[3, 4]``, which *includes* 4.

        .. versionadded:: 0.11.0

        Examples
        --------
        >>> from scipy import stats
        >>> import matplotlib.pyplot as plt

        First a basic example:

        >>> stats.binned_statistic([1, 2, 1, 2, 4], np.arange(5),
        ...                         statistic='mean',bins=3)
        (array([ 1.,  2.,  4.]), array([ 1.,  2.,  3.,  4.]),
         array([1, 2, 1, 2, 3]))

        As a second example, we now generate some random data of sailing boat
        speed as a function of wind speed, and then determine how fast our boat
        is for certain wind speeds:

        >>> windspeed = 8 * np.random.rand(500)
        >>> boatspeed = .3 * windspeed**.5 + .2 * np.random.rand(500)
        >>> bin_means, bin_edges, binnumber = stats.binned_statistic(windspeed,
        ...              boatspeed, statistic='median', bins=[1,2,3,4,5,6,7])
        >>> plt.figure()
        >>> plt.plot(windspeed, boatspeed, 'b.', label='raw data')
        >>> plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g',
        ...            lw=5, label='binned statistic of data')
        >>> plt.legend()

        Now we can use ``binnumber`` to select all datapoints with a windspeed
        below 1:

        >>> low_boatspeed = boatspeed[binnumber == 0]

        As a final example, we will use ``bin_edges`` and ``binnumber`` to make
        a plot of a distribution that shows the mean and distribution around
        that mean per bin, on top of a regular histogram and the probability
        distribution function:

        >>> x = np.linspace(0, 5, num=500)
        >>> x_pdf = stats.maxwell.pdf(x)
        >>> samples = stats.maxwell.rvs(size=10000)

        >>> bin_means, bin_edges, binnumber = stats.binned_statistic(x, x_pdf,
        ...         statistic='mean', bins=25)
        >>> bin_width = (bin_edges[1] - bin_edges[0])
        >>> bin_centers = bin_edges[1:] - bin_width/2

        >>> plt.figure()
        >>> plt.hist(samples, bins=50, normed=True, histtype='stepfilled',
        ...          alpha=0.2, label='histogram of data')
        >>> plt.plot(x, x_pdf, 'r-', label='analytical pdf')
        >>> plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g',
        ...            lw=2, label='binned statistic of data')
        >>> plt.plot((binnumber - 0.5) * bin_width, x_pdf, 'g.', alpha=0.5)
        >>> plt.legend(fontsize=10)
        >>> plt.show()

        """
        try:
            N = len(bins)
        except TypeError:
            N = 1

        if N != 1:
            bins = [np.asarray(bins, float)]

        if range is not None:
            if len(range) == 2:
                range = [range]

        self.statistic = statistic
        self._binned_statistic_dd_init([x], bins, range)

    def _binned_statistic_dd_init(self, sample,
                                  bins=10, range=None):
        """
        Compute a multidimensional binned statistic for a set of data.

        This is a generalization of a histogramdd function.  A
        histogram divides the space into bins, and returns the count
        of the number of points in each bin.  This function allows the
        computation of the sum, mean, median, or other statistic of
        the values within each bin.

        Parameters
        ----------
        sample : array_like
            Data to histogram passed as a sequence of D arrays of length N, or
            as an (N,D) array.
        bins : sequence or int, optional
            The bin specification:

              * A sequence of arrays describing the bin edges along each
                dimension.
              * The number of bins for each dimension (nx, ny, ... =bins)
              * The number of bins for all dimensions (nx=ny=...=bins).
        range : sequence, optional
            A sequence of lower and upper bin edges to be used if the
            edges are not given explicitely in `bins`. Defaults to the
            minimum and maximum values along each dimension.

        """
        known_stats = ['mean', 'median', 'count', 'sum', 'std']
        if not callable(self.statistic) and self.statistic not in known_stats:
            raise ValueError('invalid statistic %r' % (self.statistic,))

        # This code is based on np.histogramdd
        try:
            # Sample is an ND-array.
            N, self.D = sample.shape
        except (AttributeError, ValueError):
            # Sample is a sequence of 1D arrays.
            sample = np.atleast_2d(sample).T
            N, self.D = sample.shape

        self.nbin = np.empty(self.D, int)
        self.edges = self.D * [None]
        dedges = self.D * [None]

        try:
            M = len(bins)
            if M != self.D:
                raise AttributeError('The dimension of bins must be equal '
                                     'to the dimension of the sample x.')
        except TypeError:
            bins = self.D * [bins]

        # Select range for each dimension
        # Used only if number of bins is given.
        if range is None:
            smin = np.atleast_1d(np.array(sample.min(0), float))
            smax = np.atleast_1d(np.array(sample.max(0), float))
        else:
            smin = np.zeros(self.D)
            smax = np.zeros(self.D)
            for i in np.arange(self.D):
                smin[i], smax[i] = range[i]

        # Make sure the bins have a finite width.
        for i in np.arange(len(smin)):
            if smin[i] == smax[i]:
                smin[i] = smin[i] - .5
                smax[i] = smax[i] + .5

        # Create edge arrays
        for i in np.arange(self.D):
            if np.isscalar(bins[i]):
                self.nbin[i] = bins[i] + 2  # +2 for outlier bins
                self.edges[i] = np.linspace(smin[i], smax[i], self.nbin[i] - 1)
            else:
                self.edges[i] = np.asarray(bins[i], float)
                self.nbin[i] = len(self.edges[i]) + 1  # +1 for outlier bins
            dedges[i] = np.diff(self.edges[i])

        self.nbin = np.asarray(self.nbin)

        # Compute the bin number each sample falls into.
        Ncount = {}
        for i in np.arange(self.D):
            Ncount[i] = np.digitize(sample[:, i], self.edges[i])

        # Using digitize, values that fall on an edge are put in the
        # right bin.  For the rightmost bin, we want values equal to
        # the right edge to be counted in the last bin, and not as an
        # outlier.

        for i in np.arange(self.D):
            # Rounding precision
            decimal = int(-np.log10(dedges[i].min())) + 6
            # Find which points are on the rightmost edge.
            on_edge = np.where(np.around(sample[:, i], decimal) ==
                               np.around(self.edges[i][-1], decimal))[0]
            # Shift these points one bin to the left.
            Ncount[i][on_edge] -= 1

        # Compute the sample indices in the flattened statistic matrix.
        self.ni = self.nbin.argsort()
        self.xy = np.zeros(N, int)
        for i in np.arange(0, self.D - 1):
            self.xy += Ncount[self.ni[i]] * self.nbin[self.ni[i + 1:]].prod()
        self.xy += Ncount[self.ni[-1]]
        if self.statistic in ['mean', 'std', 'count']:
            self.flatcount = np.bincount(self.xy, None)
        self.result = np.empty(self.nbin.prod(), float)

    def __call__(self, values):
        """
        Parameters
        ----------
        values : array_like
            The values on which the statistic will be computed.  This must be
            the same shape as `x` in the constructor.

        Returns
        -------
        statistic : array
            The values of the selected statistic in each bin.
        bin_edges : array of dtype float
            Return the bin edges ``(length(statistic)+1)``.
        binnumber : 1-D ndarray of ints
            This assigns to each observation an integer that represents the bin
            in which this observation falls. Array has the same length as
            values.
        """

        if self.statistic == 'mean':
            self.result.fill(np.nan)
            flatsum = np.bincount(self.xy, values)
            a = self.flatcount.nonzero()
            self.result[a] = flatsum[a] / self.flatcount[a]
        elif self.statistic == 'std':
            self.result.fill(0)
            flatsum = np.bincount(self.xy, values)
            flatsum2 = np.bincount(self.xy, values ** 2)
            a = self.flatcount.nonzero()
            self.result[a] = np.sqrt(flatsum2[a] / self.flatcount[a] -
                                     (flatsum[a] / self.flatcount[a]) ** 2)
        elif self.statistic == 'count':
            self.result.fill(0)
            a = np.arange(len(self.flatcount))
            self.result[a] = self.flatcount
        elif self.statistic == 'sum':
            self.result.fill(0)
            flatsum = np.bincount(self.xy, values)
            a = np.arange(len(flatsum))
            self.result[a] = flatsum
        elif self.statistic == 'median':
            self.result.fill(np.nan)
            for i in np.unique(self.xy):
                self.result[i] = np.median(values[self.xy == i])
        elif callable(self.statistic):
            with warnings.catch_warnings():
                # Numpy generates a warnings for mean/std/... with empty list
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                old = np.seterr(invalid='ignore')
                try:
                    null = self.statistic([])
                except:
                    null = np.nan
                np.seterr(**old)
            self.result.fill(null)
            for i in np.unique(self.xy):
                self.result[i] = self.statistic(values[self.xy == i])

        # Shape into a proper matrix
        self.result = self.result.reshape(np.sort(self.nbin))
        for i in np.arange(self.nbin.size):
            j = self.ni.argsort()[i]
            self.result = self.result.swapaxes(i, j)
            self.ni[i], self.ni[j] = self.ni[j], self.ni[i]

        # Remove outliers (indices 0 and -1 for each dimension).
        core = self.D * [slice(1, -1)]
        self.result = self.result[core]

        if (self.result.shape != self.nbin - 2).any():
            raise RuntimeError('Internal Shape Error')

        return self.result


class RadialBinnedStatistic(BinnedStatistic):
    """
    Create a 1-dimensional histogram by binning a 2-dimensional
    image in radius.
    """

    def __init__(self, xsize, ysize, statistic='mean', bins=10,
                 xc=None, yc=None, rrange=None, phirange=None,
                 mask=None, cartesian=True):
        """
        Parameters:
        -----------
        xsize,ysize: int
            shape of image in pixels.  see "cartesian" parameter
            for definition of x/y.
        bins: int
            number of bins in histogram.
        xc,yc: int, optional
            location (in pixels) of origin (default: image center).
            see "cartesian" parameter for definition of x/y.
        rrange: (float, float), optional
            The lower and upper radial range of the bins, in pixels.
            If not provided, all pixel r values are included.
        phirange: (float, float), optional
            phi range to include.  Values are in the range
            (-180,180) degrees (default: no limits).  Phi is
            computed as arctan(y/x) where the meaning of y and x
            is determined by the setting of the "cartesian" parameter.
        mask: 2-dimensional np.ndarray, optional
            array of zero/non-zero values, same shape as image used
            in __call__.  zero values will be ignored.
        cartesian: bool, optional
            if True, use "cartesian" ordering, with x corresponding
            to matrix columns and y corresponding to matrix rows.
            Otherwise the opposite ("matrix" ordering). (default: True).
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
        self.expected_shape = xgrid.shape

        rpix = np.sqrt(xgrid**2 + ygrid**2)

        if phirange is not None:
            if cartesian:
                phipix = np.arctan2(ygrid, xgrid) * 180 / np.pi
            else:
                phipix = np.arctan2(xgrid, ygrid) * 180 / np.pi

        # exclude pixels this is arguably not-ideal, forcing masked
        # pixels to be outside the histogram range, but it's more
        # performant, because we don't have to iterate over the array
        # a second time on each __call__ picking out the same unmasked
        # pixels

        if rrange is None:
            rrange = (rpix.min(), rpix.max())
        excludeval = rrange[0]-1
        if phirange is not None:
            rpix[phipix < phirange[0]] = excludeval
            rpix[phipix > phirange[1]] = excludeval
        if mask is not None:
            if mask.shape != self.expected_shape:
                raise ValueError('"mask" has incorrect shape')
            rpix[mask == 0] = excludeval

        super(RadialBinnedStatistic, self).__init__(rpix.reshape(-1),
                                                    statistic,
                                                    bins=bins, range=rrange)

    def __call__(self, values):
        # check for what I believe could be a common error
        if values.shape != self.expected_shape:
            raise ValueError('"values" has incorrect shape')
        return super(RadialBinnedStatistic, self).__call__(values.reshape(-1))
