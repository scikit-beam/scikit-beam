from skbeam.core.accumulators.binned_statistic import RadialBinnedStatistic
from numpy.testing import assert_array_almost_equal
import numpy as np


class TestRadialBinnedStatistic(object):

    def setup(self):

        # Create test image - a sinc function.
        # Integrating in phi will produce sin(x)

        xsize, ysize = (900, 1024)

        xarr = np.arange(xsize)
        yarr = np.arange(ysize)
        xgrid, ygrid = np.meshgrid(yarr, xarr)
        self.rgrid = np.sqrt(xgrid**2 + ygrid**2)
        # use 100 here to make the function go through many periods
        self.oscillation_rate = 100.0
        self.image = np.sinc(self.rgrid / self.oscillation_rate)

    def testRadialBinnedStatistic(self):

        params = [[self.image.shape[0], self.image.shape[1], False],
                  [self.image.shape[1], self.image.shape[0], True]]
        mykwargs = [{'xc': 0, 'yc': 0, 'rrange': (100, 900), 'bins': 100,
                     'phirange': (5, 60)},
                    {'xc': 0, 'yc': 0, 'bins': 100}]
        # only test 60 bins where we don't have r-limits, because
        # past that the number of pixels is no longer proportional
        # to the radius
        myslice = [np.s_[:], np.s_[:60]]

        for xsize, ysize, cartesian in params:
            for kwargs, slice in zip(mykwargs, myslice):
                for stat in ['mean', 'median', 'count', 'sum', 'std']:
                    kwargs['cartesian'] = cartesian
                    radbinstat = RadialBinnedStatistic(xsize, ysize,
                                                       statistic=stat,
                                                       **kwargs)
                    binned = radbinstat(self.image)
                    binned /= binned.max()
                    binned = binned[slice]
                    centeroffset = 0.5*(radbinstat.edges[0][1] -
                                        radbinstat.edges[0][0])
                    ref = np.sin((radbinstat.edges[0][:-1] + centeroffset) /
                                 self.oscillation_rate * np.pi)[slice]
                    # can't check equality if we use normalization with
                    # current testing strategy, but at least check code runs
                    if stat is 'sum':
                        # the binned image won't be precisely equal to the
                        # analytic np.sin formula because the pixel r-values
                        # are quantized.  this is most dramatic where there
                        # are few pixels
                        assert_array_almost_equal(ref, binned, decimal=1)
        # test exception when BinnedStatistic is given array of incorrect shape
        try:
            radbinstat(self.image[:10, :10])
            raise RuntimeError('BinnedStatistic failed to raise exception'
                               ' when given an array of incorrect shape')
        except ValueError:
            pass
        # test exception when RadialBinnedStatistic is given 1D array
        try:
            RadialBinnedStatistic(xsize, ysize, mask=np.array([1, 2, 3, 4]))
            raise RuntimeError('RadialBinnedStatistic failed to raise'
                               ' when passed 1D array')
        except ValueError:
            pass
