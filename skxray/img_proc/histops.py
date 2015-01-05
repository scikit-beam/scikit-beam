# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Oct. 2013
"""
This class is designed to evaluate data set histograms by providing 
functions for easily plotting, evaluating, saving, and modifiying histograms.
"""


import numpy as np
from skxray.core import bin_edges_to_centers


def rescale_intensity_values(src_data,
                             max_final=255,
                             min_final=0,
                             out_dType='uint8'):
    """
    The purpose of this function is to allow easy conversion, scaling, or 
    expansion of data set intensity ranges for additional histogram analysis
    or data manipulation. Scaling is accomplished by converting all source
    values to 64bit float, followed by normalizing all values in the source
    data set by dividing each value by the total range of values in the data
    set. Thusly, normalized values range from 0 to 1. The normalized values
    are then multiplied by the new total range of values (as calculated from
    the specified new max and min values), and if the minimum value is not
    zero then the adjusted values are corrected to the proper min and max
    value by applying an offset


    Parameters
    ----------
    src_data : ndarray
        Specifies the data set you want to rescale
    
    max_final : float
        Specify the new maximum value for the data set. Default 
        value is 254

    min_final : float
        Specify the new minimum value for the data set. Default 
        value is 0

    out_dType : np.dtype
        Specify the desired data type for the output. The default 
        resulting data type is 'uint8'. If desired resulting data type is
        something other than 'uint8' then specify the desired data type here. 
        Recognizable options include:
            'int8'
            'int16'
            'int32'
            'int64'
            'uint16'
            'uint32'
            'uint64'
            'float16'
            'float32'
            'float64'

    Returns
    -------
    result : ndarray
        Output array can be an JxK (2D) or IxJxK (3D) numpy array
        Returns the resulting array to the designated variable
    """
    src_float = np.asarray(src_data, dtype='float64')
    max_initial = np.amax(src_float)
    min_initial = np.amin(src_float)
    range_initial = max_initial - min_initial
    range_final = max_final - min_final
    #if 'int' in str(src_data.dtype):
    #    range_initial = range_initial + 1
    scale_factor = (range_final)/(range_initial)
    normalized_data = scale_factor * (src_float - min_initial)
    scaled_data = normalized_data + min_initial
    if 'int' in out_dType:
        result = np.floor(scaled_data)
    result = result.astype(out_dType)
    return result

#dType list for vistrails wrapper
out_dType = ['int8',
             'int16',
             'int32',
             'int64',
             'uint16',
             'uint32',
             'uint64',
             'float16',
             'float32',
             'float64']


def hist_make(src_data,
              num_bins,
              pd_function=False):
    """
    This function evaluates the histogram of the source data set

    Parameters
    ----------
    src_data : ndarray
        Can be JxK or IxJxK
        Specifies the source data set from which you want to evaluate the
        histogram.

    num_bins : int
        Specify the number of bins to include in the histogram as an integer.
        
    pd_function : bool, optional
        Identify whether the histogram data should be normalized as a
        probability density histogram or not.
        Options:
            True -- Histogram data is normalized to range from 0 to 1
            False -- Histogram data reported simply as "counts" (e.g. Voxel 
                     Count)

    Returns
    -------
    hist : array
        1xN array containing all of the actual bin measurements (e.g. voxel
        counts)

    bin_avg : array
        1xN array containing the average intensity value for each bin.
        NOTE: the length of this array is equal to the length of the hist
        array

    bin_edges : array
        1xN array containing the edge values for each bin
        NOTE: the length of this array is 1 larger than the length of the
        hist array (e.g. len(bin_edges) = len(hist) + 1)
    """
    hist, bin_edges = np.histogram(src_data,
                                   bins=num_bins,
                                   density=pd_function)
    bin_avg = bin_edges_to_centers(bin_edges)
    return hist, bin_avg, bin_edges
