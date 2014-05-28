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
"""

import scipy.ndimage.filters as fltr


def gauss_fltr(src_data, 
               sigma, 
               order):
    """
    This function applies a multi-dimensional gaussian filter to the source 
    data set.

    Parameters
    ----------
    src_data : numpy array
        Specify the volume to be filtered
    
    sigma : int or int containing tuple
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
    output : numpy array
        The function returns the array containing the filtered result to the 
        specified variable.
    """
    output = fltr.gaussian_filter(src_data, 
                                  sigma, 
                                  order)
    return output


def median_fltr(src_data, 
                kernel_size):
    """
    This function applies a multi-dimensional median filter to the source data 
    set.
    
    Parameters
    ----------
    src_data : numpy array
        Specify the volume to be filtered
    
    kernel_size : tuple of kernel dimensions
        Specify the kernel size to be utilized in the filtering operation

    Returns
    -------
    output : numpy array
        The function returns the array containing the filtered result to the 
        specified variable.
    """
    output = fltr.median_filter(src_data, 
                                size=kernel_size)
    return output


def min_fltr(src_data, 
             kernel_size):
    """
    This function applies a multi-dimensional minimum filter to the source data 
    set.

    Parameters
    ----------
    src_data : numpy array
        Specify the volume to be filtered
    
    kernel_size : tuple of kernel dimensions
        Specify the kernel size to be utilized in the filtering operation

    Returns
    -------
    output : numpy array
        The function returns the array containing the filtered result to the 
        specified variable.
    """
    output = fltr.minimum_filter(src_data, 
                                 size=kernel_size)
    return output


def max_fltr(src_data, 
             kernel_size):
    """
    This function applies a multi-dimensional maximum filter to the source data 
    set.
    
    Parameters
    ----------
    src_data : numpy array
        Specify the volume to be filtered
    
    kernel_size : tuple of kernel dimensions
        Specify the kernel size to be utilized in the filtering operation

    Returns
    -------
    output : numpy array
        The function returns the array containing the filtered result to the 
        specified variable.
    """
    output = fltr.maximum_filter(src_data, 
                                 size=kernel_size)
    return output


def gradMag_gauss_fltr (src_data, 
                        sigma):
    """
    This function applies a multi-dimensional gradient magnitude filter using 
    Gaussian derivatives on the source data set.
    
    Parameters
    ----------
    src_data : numpy array
        Specify the volume to be filtered
    
    sigma : int or int tuple
        This value specifies the standard deviation for the gaussian filter and 
        can be entered as a single value, or for anisotropic filtering this 
        parameter can contain a sequence of values associated with each axis of 
        the source data.

    Returns
    -------
    output : numpy array
        The function returns the array containing the filtered result to the 
        specified variable.
    """
    output = fltr.gaussian_gradient_magnitude(src_data, 
                                              sigma)
    return output


def laplace_gauss_fltr (src_data, 
                        sigma):
    """
    This function applies a multi-dimensional Laplace filter using Gaussian 
    second derivatives on the source data set.
    
    Parameters
    ----------
    src_data : numpy array
        Specify the volume to be filtered
    
    sigma : int
        This value specifies the standard deviation for the gaussian filter and 
        can be entered as a single value, or for anisotropic filtering this 
        parameter can contain a sequence of values associated with each axis of 
        the source data.

    Returns
    -------
    output : numpy array
        The function returns the array containing the filtered result to the 
        specified variable.
    """
    output = fltr.gaussian_laplace(src_data, 
                                   sigma)
    return output


def laplace_fltr (src_data):
    """
    This function applies a multi-dimensional Laplace filter using approximate 
    second derivatives of the source data set.
    
    Parameters
    ----------
    src_data : numpy array
        Specify the volume to be filtered
    
    Returns
    -------
    output : numpy array
        The function returns the array containing the filtered result to the 
        specified variable.
    """
    output = fltr.laplace(src_data)
    return output


def percent_fltr (src_data, 
                  percentile, 
                  kernel_size):
    """
    This function applies a multi-dimensional percentile filter to the source 
    data set. The percentile filter replaces the value of the central voxel with 
    a percentile of all voxels included in the kernel. 
    NOTE: that application of the percentile filter using the 50th percentile 
    (percentile = 50) is equivalent to the median filter.
    
    Parameters
    ----------
    src_data : numpy array
        Specify the volume to be filtered
    
    percentile : int
        Specify percentile value (range of -99 to 100) 
    
    kernel_size : int tuple
        Specify the kernel size to be utilized in the filtering operation

    Returns
    -------
    output : numpy array
        The function returns the array containing the filtered result to the 
        specified variable.
    """
    output = fltr.percentile_filter(src_data, 
                                    percentile, 
                                    size=kernel_size)
    return output


def sobel_fltr (src_data):
    """
    This function computes a basic Sobel filter on the source data set. The 
    Sobel filter can be used to aid in edge, and boundary region detection.
    
    Parameters
    ----------
    src_data : numpy array
        Specify the volume to be filtered
    
    Returns
    -------
    output : numpy array
        The function returns the array containing the filtered result to the 
        specified variable.
    """
    output = fltr.sobel(src_data)
    return output

#def TV_fltr():


#def NLmeans():


def fltr_obj_dict_create (fltr_mod_name):
    filter_ops_dict = {"kernel based" : {"median" : fltr_mod_name.median_fltr, "minimum" : fltr_mod_name.min_fltr, "maximum" : fltr_mod_name.max_fltr},
                       "percentile" : fltr_mod_name.percent_fltr,
                       "gaussian" : fltr_mod_name.gauss_fltr,
                       "gradient magnitude: gaussian" : fltr_mod_name.gradMag_gauss_fltr,
                       "laplace" : fltr_mod_name.laplace_fltr,
                       "laplace: gaussian approx." : fltr_mod_name.laplace_gauss_fltr,
                       "edge detect" : {"sobel" : fltr_mod_name.sobel_fltr}
                       }
    return filter_ops_dict

