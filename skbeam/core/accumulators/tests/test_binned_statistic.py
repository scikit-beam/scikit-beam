from skbeam.core.accumulators.binned_statistic import RadialBinnedStatistic
from nose.tools import assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal
import numpy as np
import scipy.stats


class TestRadialBinnedStatistic(object):
    oscillation_rate = 10.0

    def setup(self):

        # Create test image - a sinc function.
        # Integrating in phi will produce sin(x)

        xsize, ysize = (90, 102)

        xarr = np.arange(xsize)
        yarr = np.arange(ysize)
        xgrid, ygrid = np.meshgrid(yarr, xarr)
        self.rgrid = np.sqrt(xgrid**2 + ygrid**2)
        self.image = np.sinc(self.rgrid / self.oscillation_rate)

    def testRadialBinnedStatistic(self):

        params = [[100, self.image.shape[0], self.image.shape[1], False],
                  [100, self.image.shape[1], self.image.shape[0], True]]
        mykwargs = [{'xc': 0, 'yc': 0, 'rrange': (10, 90),
                     'phirange': (5, 60)},
                    {'xc': 0, 'yc': 0}]

        for bins, xsize, ysize, cartesian in params:
            for kwargs in mykwargs:
                for stat, stat_func in [('mean', np.mean),
                                        ('median', np.median),
                                        ('count', len),
                                        ('sum', np.sum), ('std', np.std)]:

                    radbinstat = RadialBinnedStatistic(bins, xsize, ysize,
                                                       cartesian,
                                                       statistic=stat,
                                                       **kwargs)
                    radbinstat_f = RadialBinnedStatistic(bins, xsize, ysize,
                                                         cartesian,
                                                         statistic=stat_func,
                                                         **kwargs)
                    binned = radbinstat(self.image)
                    binned_f = radbinstat_f(self.image)

                    assert_array_almost_equal(binned_f, binned)
                    # can't check equality if we use normalization with
                    # current testing strategy, but at least check code runs
                    if 'phirange' not in kwargs:
                        rrange = kwargs.get('rrange', None)
                        ref, _, _ = scipy.stats.binned_statistic(
                            x=self.rgrid.ravel(),
                            values=self.image.ravel(),
                            statistic=stat,
                            range=rrange,
                            bins=bins,
                        )

                        assert_array_equal(ref, binned)

        # test exception when BinnedStatistic is given array of incorrect shape
        with assert_raises(ValueError):
            radbinstat(self.image[:10, :10])

        # test exception when RadialBinnedStatistic is given 1D array
        with assert_raises(ValueError):
            RadialBinnedStatistic(10, xsize, ysize, True,
                                  mask=np.array([1, 2, 3, 4]))
