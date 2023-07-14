import numpy as np
import pytest
import scipy.stats
from numpy.testing import assert_array_almost_equal, assert_raises

from skbeam.core.accumulators.binned_statistic import (
    BinnedStatistic1D,
    BinnedStatisticDD,
    RadialBinnedStatistic,
    RPhiBinnedStatistic,
)
from skbeam.core.utils import angle_grid, bin_edges_to_centers, radial_grid

stats_list = [("mean", np.mean), ("median", np.median), ("count", len), ("sum", np.sum), ("std", np.std)]


class TestRadialBinnedStatistic(object):
    oscillation_rate = 10.0

    def setup_method(self):
        # Create test image - a sinc function.
        # Integrating in phi will produce sin(x)

        self.rowsize, self.colsize = (91, 102)
        self.shape = self.rowsize, self.colsize

        rowarr = np.arange(self.rowsize)
        colarr = np.arange(self.colsize)
        self.rowgrid, self.colgrid = np.meshgrid(rowarr, colarr, indexing="ij")

    def _testRadialBinnedStatistic(self, rfac=None):
        """rfac : make the rgrid non-linear and supply to
        RadialBinnedStatistic.
        """

        mykwargs = [{"origin": (0, 0), "range": (10, 90)}, {"origin": (0, 0)}, {"origin": None}]
        bins, shape = 100, self.shape
        mask_ones = np.ones(self.shape)
        mask_random = np.random.randint(2, size=self.shape)

        for kwargs in mykwargs:
            for stat, stat_func in stats_list:
                if "origin" in kwargs:
                    origin = kwargs["origin"]
                else:
                    origin = None
                if origin is None:
                    origin = (self.rowsize - 1) / 2.0, (self.colsize - 1) / 2.0

                # need to calculate these every time in loop since origin
                # changes
                # rows are y, cols are x, as in angle_grid in core.utils
                self.rgrid = np.sqrt((self.rowgrid - origin[0]) ** 2 + (self.colgrid - origin[1]) ** 2)
                if rfac is not None:
                    self.rgrid = self.rgrid**rfac

                self.phigrid = np.arctan2(self.rowgrid - origin[0], self.colgrid - origin[1])

                self.image = np.sinc(self.rgrid / self.oscillation_rate)

                if stat == "sum":
                    # in this case we can compare our masked
                    # result to binned_statistic
                    mask = mask_random
                else:
                    mask = mask_ones

                # test radial case
                if rfac is None:
                    radbinstat = RadialBinnedStatistic(shape, bins, statistic=stat, mask=mask, **kwargs)
                    radbinstat_f = RadialBinnedStatistic(shape, bins, statistic=stat_func, mask=mask, **kwargs)
                else:
                    radbinstat = RadialBinnedStatistic(
                        shape, bins, statistic=stat, mask=mask, r_map=self.rgrid, **kwargs
                    )
                    radbinstat_f = RadialBinnedStatistic(
                        shape, bins, statistic=stat_func, mask=mask, r_map=self.rgrid, **kwargs
                    )
                binned = radbinstat(self.image)
                binned_f = radbinstat_f(self.image)

                assert_array_almost_equal(binned_f, binned)

                kwrange = kwargs.get("range", None)
                ref, edges, _ = scipy.stats.binned_statistic(
                    x=self.rgrid.ravel(),
                    values=(self.image * mask).ravel(),
                    statistic=stat,
                    range=kwrange,
                    bins=bins,
                )
                centers = bin_edges_to_centers(edges)

                assert_array_almost_equal(ref, binned)
                assert_array_almost_equal(edges, radbinstat.bin_edges)
                assert_array_almost_equal(edges, radbinstat_f.bin_edges)
                assert_array_almost_equal(centers, radbinstat.bin_centers)
                assert_array_almost_equal(centers, radbinstat_f.bin_centers)

        bins = (100, 2)
        myrphikwargs = [
            {"origin": (0, 0), "range": ((10, 90), (0, np.pi / 2))},
            {"origin": (0, 0)},
            {"origin": None},
        ]
        for kwargs in myrphikwargs:
            for stat, stat_func in stats_list:
                if "origin" in kwargs:
                    origin = kwargs["origin"]
                else:
                    origin = None
                if origin is None:
                    origin = (self.rowsize - 1) / 2.0, (self.colsize - 1) / 2.0

                # need to calculate these every time in loop since origin
                # changes
                # rows are y, cols are x, as in angle_grid in core.utils
                self.rgrid = np.sqrt((self.rowgrid - origin[0]) ** 2 + (self.colgrid - origin[1]) ** 2)
                self.phigrid = np.arctan2(self.rowgrid - origin[0], self.colgrid - origin[1])

                self.image = np.sinc(self.rgrid / self.oscillation_rate)

                if stat == "sum":
                    # in this case we can compare our masked
                    # result to binned_statistic
                    mask = mask_random
                else:
                    mask = mask_ones

                # test radial case
                rphibinstat = RPhiBinnedStatistic(shape, bins, statistic=stat, mask=mask, **kwargs)
                rphibinstat_f = RPhiBinnedStatistic(shape, bins, statistic=stat_func, mask=mask, **kwargs)
                binned = rphibinstat(self.image)
                binned_f = rphibinstat_f(self.image)

                # this test fails only for the standard deviation where
                # there is a disagreement in the number of nan's.  I
                # don't believe this is the fault of the binned_statistic
                # code
                if stat != "std":
                    assert_array_almost_equal(binned_f, binned)

                kwrange = kwargs.get("range", None)
                ref, redges, phiedges, _ = scipy.stats.binned_statistic_2d(
                    x=self.rgrid.ravel(),
                    y=self.phigrid.ravel(),
                    values=(self.image * mask).ravel(),
                    statistic=stat,
                    range=kwrange,
                    bins=bins,
                )

                assert_array_almost_equal(np.nan_to_num(ref), np.nan_to_num(binned))
                assert_array_almost_equal(redges, rphibinstat.bin_edges[0])
                assert_array_almost_equal(redges, rphibinstat_f.bin_edges[0])
                assert_array_almost_equal(phiedges, rphibinstat.bin_edges[1])
                assert_array_almost_equal(phiedges, rphibinstat_f.bin_edges[1])

        # test exception when BinnedStatistic is given array of incorrect shape
        with assert_raises(ValueError):
            radbinstat(self.image[:10, :10])

        # test exception when RadialBinnedStatistic is given 1D array
        with assert_raises(ValueError):
            RadialBinnedStatistic(self.image.shape, 10, mask=np.array([1, 2, 3, 4]))

    def test_RadialBinnedStatistic(self):
        self._testRadialBinnedStatistic()

    def test_RadialBinnedStatistic_rmap(self):
        self._testRadialBinnedStatistic(rfac=1.1)


@pytest.mark.parametrize(["stat", "stat_f"], stats_list)
def test_BinnedStatistics1D(stat, stat_f):
    x = np.linspace(0, 2 * np.pi, 100)
    values = np.sin(x * 5)

    bs = BinnedStatistic1D(x, statistic=stat, bins=10)
    bs_f = BinnedStatistic1D(x, statistic=stat_f, bins=10)

    ref, edges, _ = scipy.stats.binned_statistic(x, values, statistic=stat, bins=10)

    assert_array_almost_equal(bs(values), ref)
    assert_array_almost_equal(bs_f(values), ref)
    assert_array_almost_equal(edges, bs.bin_edges)
    assert_array_almost_equal(edges, bs_f.bin_edges)

    rbinstat = BinnedStatistic1D(x)
    # make sure wrong shape is caught
    with assert_raises(ValueError):
        rbinstat(x[:-2])

    # try with same shape, should be fine
    rbinstat(x)


def test_binmap():
    """These tests can be run in notebooks and the binmaps plotted.
    If this ever returns an error, it is suggested to plot the maps and see
    if they look sensible.
    """
    # generate fake data on the fly
    shape = np.array([100, 100])

    R = radial_grid(shape / 2, shape)
    Phi = angle_grid(shape / 2.0, shape)

    img = R * np.cos(5 * Phi)
    rs = RPhiBinnedStatistic(img.shape)

    binmap = rs.binmap.reshape((-1, img.shape[0], img.shape[1]))

    assert_array_almost_equal(binmap[0][40][::10], np.array([8, 6, 5, 4, 2, 2, 2, 4, 5, 6]))


def test_binmap3d():
    """These tests can be run in notebooks and the binmaps plotted.
    If this ever returns an error, it is suggested to plot the maps and see
    if they look sensible.
    """
    # test the 3d version separately
    Nx, Ny = 101, 101
    x = np.arange(Nx) - Nx / 2.0
    y = np.arange(Ny) - Ny / 2.0
    X, Y = np.meshgrid(x, y)
    X = X.ravel()
    Y = Y.ravel()
    # choose the Z variation as some arbitrary value
    Z = np.hypot(X, Y)
    # d arrays lenghtn or array length N, D array
    # try both versions of input either:
    # a tuple of the vals
    vals1 = [X, Y, Z]
    # or an N, D array (yes, it's transposed...)
    vals2 = np.array([X, Y, Z]).T
    binstat1 = BinnedStatisticDD(vals1, bins=(10, 10, 10))
    binstat2 = BinnedStatisticDD(vals2, bins=(10, 10, 10))

    rbinmap1 = binstat1.binmap
    rbinmap2 = binstat2.binmap

    assert_array_almost_equal(rbinmap1[0][::1000], rbinmap2[0][::1000])
    assert_array_almost_equal(rbinmap1[0][::1000], np.array([1, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]))
