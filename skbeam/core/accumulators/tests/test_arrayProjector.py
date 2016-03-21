from skbeam.core.accumulators.arrayProjector import RadialProjector
import numpy as np

def MakeImage(shape) :
    # Create test image - a sinc function, centered in the middle of
    # the image  
    # Integrating in phi about center, the integral will become sin(x)    

    xsize, ysize = shape
    ratio = float(ysize)/float(xsize)
    xmin, xmax = -4, 6 
    ymin, ymax = -7*ratio, 3*ratio

    xarr = np.linspace(xmin, xmax, xsize)
    yarr = np.linspace(ymin, ymax, ysize)
    xgrid, ygrid = np.meshgrid(xarr, yarr)
    rgrid = np.sqrt(xgrid**2 + ygrid**2)
    image = np.abs(np.sinc(rgrid))    
    return image

import matplotlib.pyplot as plt

def testRadialProjector():

    image = MakeImage((1024,1024))
    mask = np.ones_like(image, dtype=np.int)
    normImage = np.ones_like(image, dtype=np.int)

    ysize, xsize = image.shape
    
    angint = RadialProjector(xsize, ysize, xc=410, yc=718, rmin=0, rmax=1000, nbins=1000, norm=True)

    integral = angint(image)

    plt.plot(angint.bin_centers, integral)
    plt.show()    
