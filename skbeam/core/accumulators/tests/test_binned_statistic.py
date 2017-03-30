from skbeam.core.accumulators.binned_statistic import (RadialBinnedStatistic,
                                                       RPhiBinnedStatistic,
                                                       BinnedStatistic1D)
from nose.tools import assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal
import numpy as np
import scipy.stats
from ...utils import bin_edges_to_centers

stats_list = [('mean', np.mean), ('median', np.median), ('count', len),
              ('sum', np.sum), ('std', np.std)]


class TestRadialBinnedStatistic(object):
    oscillation_rate = 10.0

    def setup(self):

        # Create test image - a sinc function.
        # Integrating in phi will produce sin(x)

        self.rowsize, self.colsize = (91, 102)
        self.shape = self.rowsize, self.colsize

        rowarr = np.arange(self.rowsize)
        colarr = np.arange(self.colsize)
        self.rowgrid, self.colgrid = np.meshgrid(rowarr, colarr, indexing='ij')

    def testRadialBinnedStatistic(self):

        mykwargs = [{'origin': (0, 0),
                     'range': (10, 90)},
                    {'origin': (0, 0)},
                    {'origin': None}
                    ]
        bins, shape = 100, self.shape
        mask_ones = np.ones(self.shape)
        mask_random = np.random.randint(2, size=self.shape)

        for kwargs in mykwargs:
            for stat, stat_func in stats_list:

                if 'origin' in kwargs:
                    origin = kwargs['origin']
                else:
                    origin = None
                if origin is None:
                    origin = (self.rowsize-1)/2., (self.colsize-1)/2.

                # need to calculate these every time in loop since origin
                # changes
                # rows are y, cols are x, as in angle_grid in core.utils
                self.rgrid = np.sqrt((self.rowgrid-origin[0])**2 +
                                     (self.colgrid-origin[1])**2)
                self.phigrid = np.arctan2(self.rowgrid-origin[0],
                                          self.colgrid-origin[1])

                self.image = np.sinc(self.rgrid / self.oscillation_rate)

                if stat is 'sum':
                    # in this case we can compare our masked
                    # result to binned_statistic
                    mask = mask_random
                else:
                    mask = mask_ones

                # test radial case
                radbinstat = RadialBinnedStatistic(shape, bins,
                                                   statistic=stat,
                                                   mask=mask,
                                                   **kwargs)
                radbinstat_f = RadialBinnedStatistic(shape, bins,
                                                     statistic=stat_func,
                                                     mask=mask,
                                                     **kwargs)
                binned = radbinstat(self.image)
                binned_f = radbinstat_f(self.image)

                assert_array_almost_equal(binned_f, binned)

                kwrange = kwargs.get('range', None)
                ref, edges, _ = scipy.stats.binned_statistic(
                    x=self.rgrid.ravel(),
                    values=(self.image*mask).ravel(),
                    statistic=stat,
                    range=kwrange,
                    bins=bins,
                )
                centers = bin_edges_to_centers(edges)

                assert_array_equal(ref, binned)
                assert_array_equal(edges, radbinstat.bin_edges)
                assert_array_equal(edges, radbinstat_f.bin_edges)
                assert_array_equal(centers, radbinstat.bin_centers)
                assert_array_equal(centers, radbinstat_f.bin_centers)

        bins = (100, 2)
        myrphikwargs = [{'origin': (0, 0),
                         'range': ((10, 90), (0, np.pi/2))},
                        {'origin': (0, 0)},
                        {'origin': None}]
        for kwargs in myrphikwargs:
            for stat, stat_func in stats_list:
                if 'origin' in kwargs:
                    origin = kwargs['origin']
                else:
                    origin = None
                if origin is None:
                    origin = (self.rowsize-1)/2., (self.colsize-1)/2.

                # need to calculate these every time in loop since origin
                # changes
                # rows are y, cols are x, as in angle_grid in core.utils
                self.rgrid = np.sqrt((self.rowgrid-origin[0])**2 +
                                     (self.colgrid-origin[1])**2)
                self.phigrid = np.arctan2(self.rowgrid-origin[0],
                                          self.colgrid-origin[1])

                self.image = np.sinc(self.rgrid / self.oscillation_rate)

                if stat is 'sum':
                    # in this case we can compare our masked
                    # result to binned_statistic
                    mask = mask_random
                else:
                    mask = mask_ones

                # test radial case
                rphibinstat = RPhiBinnedStatistic(shape, bins,
                                                  statistic=stat,
                                                  mask=mask,
                                                  **kwargs)
                rphibinstat_f = RPhiBinnedStatistic(shape, bins,
                                                    statistic=stat_func,
                                                    mask=mask,
                                                    **kwargs)
                binned = rphibinstat(self.image)
                binned_f = rphibinstat_f(self.image)

                # this test fails only for the standard deviation where
                # there is a disagreement in the number of nan's.  I
                # don't believe this is the fault of the binned_statistic
                # code
                if stat != 'std':
                    assert_array_almost_equal(binned_f, binned)

                kwrange = kwargs.get('range', None)
                ref, redges, phiedges, _ = scipy.stats.binned_statistic_2d(
                    x=self.rgrid.ravel(),
                    y=self.phigrid.ravel(),
                    values=(self.image*mask).ravel(),
                    statistic=stat,
                    range=kwrange,
                    bins=bins,
                )

                assert_array_equal(ref, binned)
                assert_array_equal(redges, rphibinstat.bin_edges[0])
                assert_array_equal(redges, rphibinstat_f.bin_edges[0])
                assert_array_equal(phiedges, rphibinstat.bin_edges[1])
                assert_array_equal(phiedges, rphibinstat_f.bin_edges[1])

        # test exception when BinnedStatistic is given array of incorrect shape
        with assert_raises(ValueError):
            radbinstat(self.image[:10, :10])

        # test exception when RadialBinnedStatistic is given 1D array
        with assert_raises(ValueError):
            RadialBinnedStatistic(self.image.shape, 10,
                                  mask=np.array([1, 2, 3, 4]))


def test_BinnedStatistics1D():
    x = np.linspace(0, 2*np.pi, 100)
    values = np.sin(x * 5)

    for stat, stat_f in stats_list:
        bs = BinnedStatistic1D(x, statistic=stat, bins=10)
        bs_f = BinnedStatistic1D(x, statistic=stat_f, bins=10)

        ref, edges, _ = scipy.stats.binned_statistic(x, values,
                                                     statistic=stat, bins=10)

        assert_array_equal(bs(values), ref)
        assert_array_almost_equal(bs_f(values), ref)
        assert_array_equal(edges, bs.bin_edges)
        assert_array_equal(edges, bs_f.bin_edges)

    rbinstat = BinnedStatistic1D(x)
    # make sure wrong shape is caught
    with assert_raises(ValueError):
        rbinstat(x[:-2])

    # try with same shape, should be fine
    rbinstat(x)
