from skbeam.core.accumulators.bin_image import RadialBinImage, BinImage
from numpy.testing import assert_array_almost_equal
import numpy as np


class TestRadialBinImage(object):

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

    def testRadialBinImage(self):

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
                    radbinimage = RadialBinImage(xsize, ysize, nbins, norm,
                                                 **kwargs)
                    binned = radbinimage(self.image)[slice]
                    binned /= binned.max()
                    ref = np.sin(radbinimage.centers/self.oscillation_rate *
                                 np.pi)[slice]
                    # can't check equality if we use normalization with
                    # current testing strategy, but at least check code runs
                    if not norm:
                        # the binned image won't be precisely equal to the
                        # analytic np.sin formula because the pixel r-values
                        # are quantized.  this is most dramatic where there
                        # are few pixels
                        assert_array_almost_equal(ref, binned, decimal=1)
        # test exception when BinImage is given array of incorrect shape
        try:
            radbinimage(self.image[:10, :10])
            raise RuntimeError('BinImage failed to raise exception'
                               ' when given an array of incorrect shape')
        except ValueError:
            pass
        # test exception when weights and bin_values have different shape
        try:
            BinImage(np.array([1, 2]), 10, True,
                     weights=np.array([1, 2, 3]))
            raise RuntimeError('BinImage failed to raise exception'
                               ' when when weights/bin_values have'
                               ' inconsistent shapes')
        except ValueError:
            pass
        # test exception when RadialBinImage is given 1D array
        try:
            RadialBinImage(xsize, ysize, nbins, norm,
                           weights=np.array([1, 2, 3, 4]))
            raise RuntimeError('RadialBinImage failed to raise exception'
                               ' when passed 1D array')
        except ValueError:
            pass
