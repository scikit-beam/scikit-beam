from skbeam.core.accumulators.projector import RadialProjector, Projector
from numpy.testing import assert_array_almost_equal
import numpy as np


class TestRadialProjector(object):

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

    def testRadialProjector(self):

        params = [[self.image.shape[0], self.image.shape[1], False],
                  [self.image.shape[1], self.image.shape[0], True]]
        mykwargs = [{'xc': 0, 'yc': 0, 'rmin': 100, 'rmax': 900,
                     'phimin': 5, 'phimax': 60}, {'xc': 0, 'yc': 0}]
        # only test 60 bins where we don't have r-limits, because
        # past that the number of pixels is no longer proportional
        # to the radius
        myslice = [np.s_[:], np.s_[:60]]
        nbins = 100

        for norm in [True, False]:
            for xsize, ysize, cartesian in params:
                for kwargs, slice in zip(mykwargs, myslice):
                    kwargs['cartesian'] = cartesian
                    radproj = RadialProjector(xsize, ysize, nbins, norm,
                                              **kwargs)
                    projection = radproj(self.image)[slice]
                    projection /= projection.max()
                    ref = np.sin(radproj.centers/self.oscillation_rate *
                                 np.pi)[slice]
                    # can't check equality if we use normalization with
                    # current testing strategy, but at least check code runs
                    if not norm:
                        # the projection won't be precisely equal to the
                        # analytic np.sin formula because the pixel r-values
                        # are quantized.
                        assert_array_almost_equal(ref, projection, decimal=1)
        # test exception when Projector is given array of incorrect shape
        try:
            radproj(self.image[:10, :10])
            raise RuntimeError('Projector failed to raise exception'
                               ' when given an array of incorrect shape')
        except ValueError:
            pass
        # test exception when weights and bin_values have different shape
        try:
            Projector(np.array([1, 2]), 10, True,
                      weights=np.array([1, 2, 3]))
            raise RuntimeError('Projector failed to raise exception'
                               ' when when weights/bin_values have'
                               ' inconsistent shapes')
        except ValueError:
            pass
        # test exception when RadialProjector is given 1D array
        try:
            RadialProjector(xsize, ysize, nbins, norm,
                            weights=np.array([1, 2, 3, 4]))
            raise RuntimeError('RadialProjector failed to raise exception'
                               ' when passed 1D array')
        except ValueError:
            pass
