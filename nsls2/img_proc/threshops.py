# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Oct. 2013
"""
This module contains tools for thresholding data sets.
"""
"""
REVISION LOG: (FORMAT: "PROGRAMMER INITIALS: DATE -- RECORD")
GCI: 2/11/2014 -- Modifying documentation of the package functions for 
		   inclusion in the bulk module pull to GITHUB
GCI: 2/19/2014 -- Renaming module from C4_threshops to threshops.py
      Updating documentation to docstring convention.
      Changed module back from class structure to a simple module containing 
      thresholding function definitions.
"""

import numpy as np
import skimage.filter as sk
#TODO: Need to make a QT widgit that is a volume viewer that also includes a 
#slider for BOTH maximum and minimum intensity for "interactive" thresholding. 
#This widgit should then also link back to the bounded thresholding tool and 
#contain a "THRESHOLD VOLUME" button which then executes the threshold and 
#creates a binary volume of the result.

def thresh_globalGT(src_data, thresh_value):
    """
    This function is a basic thresholding operation allowing for isolation, or 
    selection, of all voxels having an intensity value greater than the 
    specificed threshold value. The result is a binary array of equal size 
    containing voxel values of either 0 or 1.

    Parameters
    ----------
    src_data : numpy array
        Specify the volume to be thresholded
    
    thresh_value : scalar of same dtype as src_data
        Specify the threshold value, above which contains the desired voxel 
        values to be isolated.
    
    Returns
    -------
    output : binary numpy array
        The function returns a binary array where all voxels with values equal 
        to 1 correspond to voxels within the identified threshold range.
    """
    output = (src_data > thresh_value).astype(np.uint8)
    print 'Thresholding Operation successful:'
    print "Selected Threshold Value: " + str(thresh_value)
    print ("Isolated regions in output data set had a measured intensity " +
           "GREATER THAN the selected threshold value.")
    return output


def thresh_globalLT(src_data, thresh_value):
    """
    This function is a basic thresholding operation allowing for isolation, or 
    selection, of all voxels having an intensity value less than the specificed 
    threshold value. The result is a binary array of equal size containing 
    voxel values of either 0 or 1.

    Parameters
    ----------
    src_data : numpy array
        Specify the volume to be thresholded

    thresh_value : scalar of same dtype as src_data
        Specify the threshold value, below which contains the desired voxel 
        values to be isolated.

    Returns
    -------
    output : binary numpy array
        The function returns a binary array where all voxels with values equal 
        to 1 correspond to voxels within the identified threshold range.
    """
    output = (src_data < thresh_value).astype(np.uint8)
    print 'Thresholding Operation successful:'
    print "Selected Threshold Value: " + str(thresh_value)
    print ("Isolated regions in output data set had a measured intensity LESS" +
           " THAN the selected threshold value.")
    return output


def thresh_bounded(src_data, thresh_value_MIN, thresh_value_MAX):
    """
    This function is a basic thresholding operation allowing for isolation, or 
    selection, of all voxels having an intensity value between the specificed 
    threshold value boundaries. The result is a binary array of equal size 
    containing voxel values of either 0 or 1.

    Parameters
    ----------
    src_data : numpy array
        Specify the volume to be thresholded.

    thresh_value_MIN : scalar of same dtype as src_data
        Specify the lower threshold boundary.

    thresh_value_MAX : scalar of same dtype as src_data
        Specify the upper threshold boundary.

    Returns
    -------
    output : binary numpy array
        The function returns a binary array where all voxels with values equal 
        to 1 correspond to voxels within the identified threshold range.
    """
    vox_abv_minVal = (thresh_value_MIN <= src_data).astype(np.uint8)
    vox_abv_maxVal = (thresh_value_MAX <= src_data).astype(np.uint8)
    output = vox_abv_minVal - vox_abv_maxVal
    print 'Thresholding Operation successful:'
    print "Selected Threshold Values:"
    print "Minimum Threshold Value: " + str(thresh_value_MIN)
    print "Maximum Threshold Value: " + str(thresh_value_MAX)
    print ("Isolated regions in output data set had a measured intensity " +
           "values between the two selected threshold points.")
    return output

def thresh_adapt (src_data, kernel, filter_type='gaussian'):
    """
    Filter type options:
         method : {'generic', 'gaussian', 'mean', 'median'}, optional
    Method used to determine adaptive threshold for local neighbourhood in
    weighted mean image.

    * 'generic': use custom function (see `param` parameter)
    * 'gaussian': apply gaussian filter (see `param` parameter for custom\
    sigma value)
    * 'mean': apply arithmetic mean filter
    * 'median': apply median rank filter
    """
    output_data = sk.threshold_adaptive(src_data, kernel, filter_type, offset=0, param=None)
    return output_data

def thresh_otsu (src_data):
    data_max = np.amax(src_data)
    data_min = np.amin(src_data)
    if (data_max-data_min) > 1000:
        num_bins = (data_max - data_min)/2
    else:
        num_bins = (data_max - data_min)
    thresh_value = sk.threshold_otsu(src_data, num_bins)
    output_data = thresh_globalGT(src_data, thresh_value)
    return output_data, thresh_value
    
def thresh_yen (src_data):
    data_max = np.amax(src_data)
    data_min = np.amin(src_data)
    if (data_max-data_min) > 1000:
        num_bins = (data_max - data_min)/2
    else:
        num_bins = (data_max - data_min)
    thresh_value = sk.threshold_otsu(src_data, num_bins)
    output_data = thresh_globalGT(src_data, thresh_value)
    return output_data, thresh_value
    
def thresh_isodata (src_data):
    data_max = np.amax(src_data)
    data_min = np.amin(src_data)
    if (data_max-data_min) > 1000:
        num_bins = int((data_max - data_min)/2)
    else:
        num_bins = int((data_max - data_min))
    thresh_value = sk.threshold_isodata(src_data, num_bins)
    output_data = thresh_globalGT(src_data, thresh_value)
    return output_data, thresh_value

    
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#TODO Add simple multi thresholding tool
#TODO Add auto thresholding methods including:
#       1st derivative
#       entropy
#       moments
#       factorization
#       OTHERS...
    #MULTI-THRESHOLDING
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # def thresh_multi(self, src_data):
        # output = not src_data
        # print 'Operation successful: ' + "NOT " + src_data
        # return output
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #FIRST DERIVATIVE THRESHOLDING
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # def thresh_firstDerivative(self, op_type='offset', offset_image=0):
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
