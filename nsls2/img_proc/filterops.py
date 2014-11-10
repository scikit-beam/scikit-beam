# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Oct. 2013
"""
This class contains functions used for filtering images and volume data sets.
The basic filters included in this class utilize the filter set included in
scipy.ndimage.filters. More advanced filtering operations will be added 
shortly.
"""
"""
REVISION LOG: (FORMAT: "PROGRAMMER INITIALS: DATE -- RECORD")
8GCI: 2/11/2014 -- Modifying documentation of the package functions for 
		   inclusion in the bulk module pull to GITHUB
GCI: 2/19/2014 -- Changed file name from C5_filterops.py to filterops.py
      Updated documentation to docstring format.
      Converted back from class structure to basic module containing function 
      definitions.
GCI: 9/19/14 -- Scipy filtering tools converted to direct imports,
and documentation for the filtering functions is reassigned to the
documentation I originally wrote for the functions. This doc conversion is
necessary in order to fit the autowrap format conventions required for
incorporation into VisTrails
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

def TV_fltr():
    pass


def NLmeans():
    pass

