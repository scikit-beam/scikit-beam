from skbeam.core.accumulators.binned_statistic import RadialBinnedStatistic
from numpy.testing import assert_array_almost_equal
import numpy as np


class TestRadialBinnedStatistic(object):

    def setup(self):

        # Create test image - a sinc function.
        # Integrating in phi will produce sin(x)

        rowsize, colsize = (900, 1024)

        rowarr = np.arange(rowsize)
        colarr = np.arange(colsize)
        rowgrid, colgrid = np.meshgrid(rowarr, colarr, indexing='ij')
        self.rgrid = np.sqrt(rowgrid**2 + colgrid**2)
        # use 100 here to make the function go through many periods
        self.oscillation_rate = 100.0
        self.image = np.sinc(self.rgrid / self.oscillation_rate)

    def testRadialBinnedStatistic(self):

        mykwargs = [{'rowc': 0, 'colc': 0, 'rrange': (100, 900),
                     'phirange': (5*np.pi/180., 60*np.pi/180.)},
                    {'rowc': 0, 'colc': 0}]
        # only test 60 bins where we don't have r-limits, because
        # past that the number of pixels is no longer proportional
        # to the radius
        myslice = [np.s_[:], np.s_[:60]]

        for kwargs, slice in zip(mykwargs, myslice):
            for stat in ['mean', 'median', 'count', 'sum', 'std']:
                radbinstat = RadialBinnedStatistic(100, self.image.shape[0],
                                                   self.image.shape[1],
                                                   statistic=stat,
                                                   **kwargs)
                binned = radbinstat(self.image)
                binned /= binned.max()
                binned = binned[slice]
                centeroffset = 0.5*(radbinstat.bin_edges[0][1] -
                                    radbinstat.bin_edges[0][0])
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
            RadialBinnedStatistic(10,
                                  self.image.shape[0], self.image.shape[1],
                                  mask=np.array([1, 2, 3, 4]))
            raise RuntimeError('RadialBinnedStatistic failed to raise'
                               ' when passed 1D array')
        except ValueError:
            pass
