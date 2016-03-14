from skbeam.core.accumulators.AngularIntegrator import AngularIntegrator
import numpy as np
from time import time
import matplotlib.pyplot as plt


import matplotlib
import matplotlib.pyplot as plt

def drawImage(arr, img_range=None, amp_range=None, figsize=(10,10)) :    # range = (left, right, low, high), amp_range=(zmin,zmax)
    fig = plt.figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='w', frameon=True)
    fig.subplots_adjust(left=0.05, bottom=0.03, right=0.98, top=0.98, wspace=0.2, hspace=0.1)
    figAxes = fig.add_subplot(111)
    #figAxes = fig.add_axes([0.15, 0.06, 0.78, 0.21])
    imAxes = figAxes.imshow(arr, origin='upper', interpolation='nearest', aspect='auto', extent=img_range)
    if amp_range != None : imAxes.set_clim(amp_range[0],amp_range[1])
    colbar = fig.colorbar(imAxes, pad=0.03, fraction=0.04, shrink=1.0, aspect=40, orientation='horizontal')

#------------------------------

def drawGraph(x,y) : 
    fig = plt.figure(figsize=(6,4), dpi=80, facecolor='w', edgecolor='w', frameon=True)
    #fig.subplots_adjust(left=0.05, bottom=0.03, right=0.98, top=0.98, wspace=0.2, hspace=0.1)
    #figAxes = fig.add_subplot(111)
    ax = fig.add_axes([0.15, 0.10, 0.78, 0.86])
    ax.plot(x,y,'b-')

#------------------------------

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

def testAngularIntegrator():

    image = np.ones([1000,1000])
    mask  = np.ones_like(image, dtype=np.int)

    print 'image.shape =', image.shape
    
    ysize, xsize = image.shape
    
    t0_sec = time()
    angint = AngularIntegrator()
    angint.setParameters(xsize, ysize, xc=1005, yc=690, rmin=0, rmax=1000, nbins=1000, mask=mask)
    #angint.setParameters(xsize, ysize, rmin=1, rmax=1200, nbins=200, mask=mask)
    print "Time consumed for indexing = %.3f sec" % (time()-t0_sec)

    plt.imshow(angint.getRBinIndexMap())
    plt.show()

    plt.imshow(image)
    plt.show()

    t0_sec = time()
    bincent, integral = angint.getRadialHistogramArrays(image)
    print "Time consumed for intensity binning = %.3f sec" % (time()-t0_sec)

    plt.plot(bincent, integral)
    plt.show()    
