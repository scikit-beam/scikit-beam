# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Oct. 2013
"""
This module contains functions used for filtering images and volume data sets.
The basic filters included in this module utilize the filter set included in
scipy.ndimage.filters. More advanced filtering operations will be added 
shortly.

Some tools are being added directly from scipy.ndimage for use in the skxray
tool set that is included for image processing in VisTrails. These tools are
wrappoed into the complete toolkit through the API. The tools from
scipy.ndimage include:
    gaussian_filter,
    median_filter,
    minimum_filter,
    maximum_filter,
    gaussian_gradient_magnitude,
    gaussian_laplace,
    laplace,
    percentile_filter,
    sobel, prewitt,
    rank_filter
"""


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


