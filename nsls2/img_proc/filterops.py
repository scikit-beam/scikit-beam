# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Oct. 2013
"""
This module contains functions used for filtering images and volume data sets.
The basic filters included in this module utilize the filter set included in
scipy.ndimage.filters. More advanced filtering operations will be added 
shortly.
"""

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import median_filter
from scipy.ndimage.filters import minimum_filter
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.filters import gaussian_gradient_magnitude
from scipy.ndimage.filters import gaussian_laplace
from scipy.ndimage.filters import laplace
from scipy.ndimage.filters import percentile_filter
from scipy.ndimage.filters import sobel
from scipy.ndimage.filters import prewitt
from scipy.ndimage.filters import rank_filter

#Need additional work in order to implement in vistrails since the function
#to apply needs to be defined in order to use these filters.
# from scipy.ndimage.filters import generic_filter
# from scipy.ndimage.filters import generic_gradient_magnitude
# from scipy.ndimage.filters import generic_laplace
# from scipy.ndimage.filters import uniform_filter
# from scipy.ndimage.fourier import fourier_ellipsoid
# from scipy.ndimage.fourier import fourier_gaussian
# from scipy.ndimage.fourier import fourier_shift
# from scipy.ndimage.fourier import fourier_uniform


