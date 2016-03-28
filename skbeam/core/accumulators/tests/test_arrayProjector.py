from skbeam.core.accumulators.arrayProjector import RadialProjector
import numpy as np

class TestRadialIntegrator(object):

    def setup(self):

        # Create test image - a sinc function, centered in the middle of the image
        # Integrating in phi about center, the integral will become sin(x)    

        xsize, ysize = (1024, 1024)

        xarr = np.arange(xsize)
        yarr = np.arange(ysize)
        xgrid, ygrid = np.meshgrid(xarr, yarr)
        self.rgrid = np.sqrt(xgrid**2 + ygrid**2)
        self.image = np.sinc(self.rgrid / 100.)

    def testRadialProjector(self):

        ysize, xsize = self.image.shape
        radproj = RadialProjector(xsize, ysize, xc=0, yc=0, rmin=0.1, rmax=1000, nbins=100, norm=False)

        projection = radproj(self.image)
        projection /= projection.max()
        print radproj.bin_centers
        ref = np.sin(radproj.bin_centers/100.*np.pi)

        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(projection)
        plt.plot(ref)
        plt.show()
