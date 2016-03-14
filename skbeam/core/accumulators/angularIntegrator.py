#!/usr/bin/env python
#------------------------------

""" 
Integrate a 2D image over angles.

Usage::

    # Initialization of object and its parameters
    ai = AngularIntegrator(rows, cols, xc=cols/2, yc=rows/2, rmin=rmin, rmax=rmax, nbins=nbins, mask=mask, phimin=-180, phimax=180)

    # do angular integration for each input image and return array of bin centers and associated normalized intensities
    bins, intensity = ai.getRadialHistogramArrays(img)

    # not setting the "mask" parameter will leave the angular integration unnormalized

author: Mikhail Dubrovin, Christopher O'Grady
"""

import math
import numpy as np

def valueToIndexProtected(V, VRange, phimin, phimax, phipixind) :
    """Input: V - numpy array of values,
              VRange - Vmin, Vmax, contains the binning parameters Vmin, Vmax, and Nbins
       Output: Array of indexes from 0 to Nbins (Nbins+1 index) with shape of V.
       The last index Nbins is used for overflow and underflow.
    """
    Vmin, Vmax, Nbins = VRange
    Nbins1 = int(Nbins)-1
    factor = float(Nbins) / float(Vmax-Vmin)
    indarr = np.int32( factor * (V-Vmin) )
    return np.select([phipixind<phimin, phipixind>phimax, V==Vmax, indarr<0, indarr>Nbins1], [0, 0, Nbins1, 0, 0], default=indarr)

#------------------------------

def divideArraysSafely(num, den) :
    """Per element divides numpy arrays result = num/den. Protected for 0 values. Arrays should have the same size."""
    if num.shape != den.shape :
        print 'divideArraysSafely: non-equal array shapes for numerator and denumerator: ', num.shape, den.shape
    num_corr =  np.select([den<1], [0], default=num)    
    den_corr =  np.select([den<1], [1], default=den)    
    return num_corr/den_corr

#------------------------------

class AngularIntegrator :
    """Angular integration of a 2D numpy array"""
    
    def __init__(self, rows, columns, xc=None, yc=None, rmin=None, rmax=None, nbins=None, mask=None, phimin=-180, phimax=180):
        """Sets image, radial histogram parameters, and mask if proper normalization on number of actual pixels is desired
        """   
        # agregate input parameters
        if mask is None:
            self.mask = np.ones([rows,columns])
        else:
            self.mask = mask

        # flip these to make it more intuitive for users who will use
        # this with arrays (matrices).  numpy.meshgrid (version 1.6) uses
        # cartesian (non-matrix) coordinates, so we need to do the flip

        self.xsize   = columns
        self.ysize   = rows

        self.xc    = self.xsize/2                       if xc    is None else xc
        self.yc    = self.ysize/2                       if yc    is None else yc
        self.rmin  = 0                                  if rmin  is None else rmin
        self.rmax  = math.sqrt(self.xc**2 + self.yc**2) if rmax  is None else rmax
        self.nbins = int((self.rmax - self.rmin)/2)     if nbins is None else nbins

        self.center = (self.xc, self.yc)
        self.rrange = (self.rmin, self.rmax, self.nbins)

        # produce all index arrays
        x = np.arange(self.xsize) - self.xc
        y = np.arange(self.ysize) - self.yc
        xgrid, ygrid = np.meshgrid(x,y)
 
        half_bin = 0.5*(self.rmax-self.rmin)/self.nbins
        self.bincent = np.linspace(self.rmin+half_bin, self.rmax-half_bin, self.nbins)

        self.rpixind  = np.sqrt(xgrid**2 + ygrid**2)

        # flip y here so that the angles correspond well to matplotlib plots.
        phipixind  = np.arctan2(-ygrid,xgrid) * 180 / np.pi

        self.rbinind  = valueToIndexProtected(self.rpixind, self.rrange, phimin, phimax, phipixind)

        if self.rbinind.shape!=self.mask.shape:
            raise Exception('self.rbinind.shape != mask.shape')
        self.binstat = np.bincount(self.rbinind.flatten(), weights=self.mask.flatten(), minlength=self.rrange[2])

    def getRadialHistogramArrays(self, image, norm=True):
        """Fills radial histogram with image intensities and optionally normalize by number of pixels
           and returns two arrays with radial bin centers and integrated (normalized) intensities. 
        """   
        if self.rbinind.shape!=image.shape:
            raise Exception('self.rbinind.shape != image.shape')
        w = image*self.mask # should avoid multiply by 1 here in the case we don't want to normalize
        bin_integral = np.bincount(self.rbinind.flatten(), weights=w.flatten(), minlength=self.rrange[2])

        if norm:
            return self.bincent, divideArraysSafely(bin_integral, self.binstat)
        else:
            return self.bincent, bin_integral

    def getRPixelIndexMap(self) :
        return self.rpixind


    def getRBinIndexMap(self) :
        return self.rbinind
