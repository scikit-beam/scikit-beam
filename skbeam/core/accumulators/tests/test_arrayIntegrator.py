from skbeam.core.accumulators.arrayIntegrator import AngularIntegrator
import numpy as np

def MakeImage(shape=(1024,1024)) :
    # Create test image - a sinc function, centered in the middle of
    # the image  
    # Integrating in phi about center, the integral will become sin(x)    
    print "Creating test image",

    xsize, ysize = shape
    ratio = float(ysize)/float(xsize)
    print 'ratio = ', ratio
    xmin, xmax = -4, 6 
    ymin, ymax = -7*ratio, 3*ratio

    print '\nxmin, xmax, xsize = ', xmin, xmax, xsize
    print '\nymin, ymax, ysize = ', ymin, ymax, ysize

    xarr = np.linspace(xmin, xmax, xsize)
    yarr = np.linspace(ymin, ymax, ysize)
    xgrid, ygrid = np.meshgrid(xarr, yarr)
    rgrid = np.sqrt(xgrid**2 + ygrid**2)
    image = np.abs(np.sinc(rgrid))    
    return image

import matplotlib.pyplot as plt

def testAngularIntegrator():

    image = MakeImage()
    mask = np.ones_like(image, dtype=np.int)
    normImage = np.ones_like(image, dtype=np.int)

    ysize, xsize = image.shape
    
    angint = AngularIntegrator(xsize, ysize, xc=410, yc=718, rmin=0, rmax=1000, nbins=1000)

    bincent, integral = angint.histogram(image,norm=True)

    plt.plot(bincent, integral)
    plt.show()    
