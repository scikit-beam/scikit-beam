from skbeam.core.accumulators.arrayProjector import RadialProjector
from numpy.testing import assert_array_almost_equal
import numpy as np

class TestRadialIntegrator(object):

    def setup(self):

        # Create test image - a sinc function.
        # Integrating in phi will produce sin(x)    

        xsize, ysize = (1000, 1024)

        xarr = np.arange(xsize)
        yarr = np.arange(ysize)
        xgrid, ygrid = np.meshgrid(xarr, yarr)
        self.rgrid = np.sqrt(xgrid**2 + ygrid**2)
        self.image = np.sinc(self.rgrid / 100.) # use 100 here to make the function go through many periods

    def testRadialProjector(self):

        ysize, xsize = self.image.shape
        radproj = RadialProjector(xsize, ysize, xc=0, yc=0, rmin=100, rmax=900, nbins=100, norm=False)

        projection = radproj(self.image)
        projection /= projection.max()
        ref = np.sin(radproj.bin_centers/100.*np.pi)

        from matplotlib import pyplot as plt
        #plt.imshow(self.image)
        #plt.show()
        plt.plot(projection)
        plt.plot(ref)
        plt.show()

        # the projection won't be precisely equal to the analytic np.sin formula because
        # the pixel r-values are quantized.
        assert_array_almost_equal(ref, projection, decimal=2)
