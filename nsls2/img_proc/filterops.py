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
#
gaussian_filter.__doc__ = (
    """
    This function applies a multi-dimensional gaussian filter to the source
    data set.

    Parameters
    ----------
    src_data : ndarray
        Specify the volume to be filtered

    sigma : tuple
        This value specifies the standard deviation for the gaussian filter and
        can be entered as a single value, or for anisotropic filtering this
        parameter can contain a sequence of values associated with each axis
        of the source data.

    order : int
        This value specifies the derivative order of the gaussian kernel which
        will be used to convolve the source data. Four options are available,
        including:
            0 -- Convolution with a Gaussian kernel
            1 -- Convolution with the first derivative of the Gaussian kernel
            2 -- Convolution with the second derivative of the Gaussian kernel
            3 -- Convolution with the third derivative of the Gaussian kernel

    Returns
    -------
    output : ndarray
        The function returns the array containing the filtered result to the
        specified variable.
    """
)

median_filter.__doc__ = (
    """
    This function applies a multi-dimensional median filter to the source data
    set.
    Parameters
    ----------
    input : array_like
        Input array to filter.

    size : tuple, optional
        See footprint, below

    footprint : array, optional
        Either size or footprint must be defined. size gives the shape that is
        taken from the input array, at every element position, to define the
        input to the filter function. footprint is a boolean array that
        specifies (implicitly) a shape, but also which of the elements within
        this shape will get passed to the filter function. Thus size=(n,m) is
        equivalent to footprint=np.ones((n,m)). We adjust size to the number
        of dimensions of the input array, so that, if the input array is
        shape (10,10,10), and size is 2, then the actual size used is (2,2,2).

    output : array, optional
        The output parameter passes an array in which to store the filter
        output.

    mode : string, optional
        Options
            'reflect', 'constant', 'nearest', 'mirror', 'wrap'
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
        Default is 'reflect'

    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.
        Default is 0.0
    origin : scalar, optional
        The origin parameter controls the placement of the filter.
        Default 0.0.
    Returns
    -------
    median_filter : ndarray
        The function returns the array containing the filtered result to the
        specified variable.
    """)

minimum_filter.__doc__ = (
    """
    This function applies a multi-dimensional minimum filter to the source data
    set.

    Parameters
    ----------
    src_data : ndarray
        Specify the volume to be filtered

    kernel_size : tuple
        Specify the kernel size to be utilized in the filtering operation

    Returns
    -------
    output : ndarray
        The function returns the array containing the filtered result to the
        specified variable.
    """
)

maximum_filter.__doc__ = (
    """
    This function applies a multi-dimensional maximum filter to the source data
    set.

    Parameters
    ----------
    src_data : ndarray
        Specify the volume to be filtered

    kernel_size : tuple
        Specify the kernel size to be utilized in the filtering operation

    Returns
    -------
    output : ndarray
        The function returns the array containing the filtered result to the
        specified variable.
    """
)

gaussian_gradient_magnitude.__doc__ = (
    """
    This function applies a multi-dimensional gradient magnitude filter using
    Gaussian derivatives on the source data set.

    Parameters
    ----------
    src_data : ndarray
        Specify the volume to be filtered

    sigma : tuple
        This value specifies the standard deviation for the gaussian filter and
        can be entered as a single value, or for anisotropic filtering this
        parameter can contain a sequence of values associated with each axis of
        the source data.

    Returns
    -------
    output : ndarray
        The function returns the array containing the filtered result to the
        specified variable.
    """
)

gaussian_laplace.__doc__ = (
    """
    This function applies a multi-dimensional Laplace filter using Gaussian
    second derivatives on the source data set.

    Parameters
    ----------
    src_data : ndarray
        Specify the volume to be filtered

    sigma : int
        This value specifies the standard deviation for the gaussian filter and
        can be entered as a single value, or for anisotropic filtering this
        parameter can contain a sequence of values associated with each axis of
        the source data.

    Returns
    -------
    output : ndarray
        The function returns the array containing the filtered result to the
        specified variable.
    """
)

laplace.__doc__ = (
    """
    This function applies a multi-dimensional Laplace filter using approximate
    second derivatives of the source data set.

    Parameters
    ----------
    src_data : ndarray
        Specify the volume to be filtered

    Returns
    -------
    output : ndarray
        The function returns the array containing the filtered result to the
        specified variable.
    """
)

percentile_filter.__doc__ = (
    """
    This function applies a multi-dimensional percentile filter to the source
    data set. The percentile filter replaces the value of the central voxel with
    a percentile of all voxels included in the kernel.
    NOTE: that application of the percentile filter using the 50th percentile
    (percentile = 50) is equivalent to the median filter.

    Parameters
    ----------
    src_data : ndarray
        Specify the volume to be filtered

    percentile : int
        Specify percentile value (range of -99 to 100)

    kernel_size : tuple
        Specify the kernel size to be utilized in the filtering operation

    Returns
    -------
    output : ndarray
        The function returns the array containing the filtered result to the
        specified variable.
    """
)

sobel.__doc__ = (
    """
    This function computes a basic Sobel filter on the source data set. The
    Sobel filter can be used to aid in edge, and boundary region detection.

    Parameters
    ----------
    src_data : ndarray
        Specify the volume to be filtered

    Returns
    -------
    output : ndarray
        The function returns the array containing the filtered result to the
        specified variable.
    """
)


def TV_fltr():
    pass


def NLmeans():
    pass

# def fltr_obj_dict_create (fltr_mod_name):
# filter_ops_dict = {"kernel based" : {"median" : fltr_mod_name.median_fltr,
# "minimum" : fltr_mod_name.min_fltr, "maximum" : fltr_mod_name.max_fltr},
#                        "percentile" : fltr_mod_name.percent_fltr,
#                        "gaussian" : fltr_mod_name.gauss_fltr,
#                        "gradient magnitude: gaussian" : fltr_mod_name.gradMag_gauss_fltr,
#                        "laplace" : fltr_mod_name.laplace_fltr,
#                        "laplace: gaussian approx." : fltr_mod_name.laplace_gauss_fltr,
#                        "edge detect" : {"sobel" : fltr_mod_name.sobel_fltr}
#                        }
#     return filter_ops_dict
#
