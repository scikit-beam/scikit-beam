#!/usr/bin/env python

import math
import numpy as np

"""
authors: Mikhail Dubrovin, TJ Lane, Christopher O'Grady
"""

class ArrayProjector(object):

    def __init__(self, bin_values, weights=None, nbins=101, norm=True):
        """
        Parameters
        ----------
        bin_values : np.ndarray
            For each pixel, this is the value of that determines which bin
            a pixel's intensity will contribute to
        weights : np.ndarray
            A weight for each pixel with the same shape as bin_values (can be zero to ignore a pixel)
        nbins : int
            The number of bins to employ. If `None` guesses a good value.
        """

        self.bin_values = bin_values
        self.weights = weights
        self.nbins = nbins
        self.norm = norm

        self.binvalRange = self.bin_values.max() - self.bin_values.min()
        self.bin_width = self.binvalRange / (float(nbins) - 1)

        self._bin_assignments = np.floor( (self.bin_values - self.bin_values.min()) / self.bin_width ).astype(np.int32)
        if weights is None:
            self._normalization_array = (np.bincount( self._bin_assignments.flatten() ) \
                                             + 1e-100).astype(np.float)
        else:
            self._normalization_array = (np.bincount( self._bin_assignments.flatten(), weights=self.weights.flatten() ) \
                                             + 1e-100).astype(np.float)

        assert self.nbins >= self._bin_assignments.max() + 1, 'incorrect bin assignments'
        self._normalization_array = self._normalization_array[:self.nbins]

        return

    def __call__(self, image):
        """
        Bin pixel intensities.
        
        Parameters
        ----------            
        image : np.ndarray
            The intensity at each pixel
        Returns
        -------
        bin_values : np.ndarray
            The average intensity in the bin
        """

        if not (image.shape == self.bin_values.shape):
            raise ValueError('`image` and `bin_values` must have the same shape',image.shape,self.bin_values.shape)

        if self.weights is None:
            weights = image.flatten()
        else:
            weights = image.flatten() * self.weights.flatten()
            if not (image.shape == self.weights.shape):
                raise ValueError('`image` and `weights` must have the same shape')

        bin_values = np.bincount(self._bin_assignments.flatten(), weights=weights)

        if self.norm:
            bin_values /= self._normalization_array

        assert bin_values.shape[0] == self.nbins

        return bin_values

    @property
    def bin_centers(self):
        """
        Returns:
        --------
        bin_centers : ndarray, float
            The center of each bin.
        """
        return (np.arange(self.nbins) + 0.5) * self.bin_width + self.bin_values.min()

class RadialProjector(ArrayProjector) :
    """ 
    Project a 2D image onto a radial axis
    """
    
    def __init__(self, rows, columns, xc=None, yc=None, rmin=None, rmax=None, phimin=None, phimax=None, nbins=101, weights=None, norm=True):
        """
        Parameters:
        -----------
        rows,columns:  shape of image to be projected
        xc,yc:         location (in pixels) of origin (defaults to center of image)
        rmin,rmax:     radial range to include in projection, in pixels
        phimin,phimax: phi range to include in projection, in degrees
        
        """   
        # flip these to make it more intuitive for users who will use
        # this with arrays (matrices).  numpy.meshgrid (version 1.6) uses
        # cartesian (non-matrix) coordinates, so we need to do the flip

        xsize   = columns
        ysize   = rows

        xc    = xsize/2                  if xc    is None else xc
        yc    = ysize/2                  if yc    is None else yc
        rmin  = 0                        if rmin  is None else rmin
        rmax  = math.sqrt(xc**2 + yc**2) if rmax  is None else rmax

        # produce all index arrays
        x = np.arange(xsize) - xc
        y = np.arange(ysize) - yc
        xgrid, ygrid = np.meshgrid(x,y)
 
        rpix  = np.sqrt(xgrid**2 + ygrid**2)
        # flip y here so that the angles correspond well to matplotlib plots.
        phipix = np.arctan2(-ygrid,xgrid) * 180 / np.pi
        
        if weights is None and (None in (phimin,phimax,rmin,rmax)):
            weights = np.ones((rows,columns))
        if phimin is not None:
            weights[phipix<phimin] = 0
        if phimax is not None:
            weights[phipix>phimax] = 0
        if rmin is not None:
            weights[rpix<rmin] = 0
        if rmax is not None:
            weights[rpix>rmax] = 0

        super(RadialProjector,self).__init__(rpix, weights, nbins=nbins, norm=norm)

    def __call__(self,image):
        """
        Bin pixel intensities.
        
        Parameters
        ----------            
        image : np.ndarray
            The intensity at each pixel
        Returns
        -------
        bin_values : np.ndarray
            The average intensity in each bin
        """
        return super(RadialProjector,self).__call__(image)
