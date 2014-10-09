# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Oct. 2013
"""
This class is designed to evaluate data set histograms by providing 
functions for easily plotting, evaluating, saving, and modifiying histograms.
"""
"""
REVISON LOG: (FORMAT: "PROGRAMMER INITIALS: DATE -- RECORD")
GCI: 2/11/2014 -- Modifying documentation of the package functions for
    inclusion in the bulk module pull to GITHUB
GCI: 2/19/2014 -- Updating module to docstring format.
    Changed filename from C3_histops.py to histops.py
    TODO Finish Adding function for saving histogram data as a CSV file or
    HDF5
    data set.
    TODO Finish adding function for loading previously saved histogram data.
GCI: 2/28/2014 -- Added the functions:
    hist_makeData: which writes histogram data to the HDF5 file
    hist_makeCSV: which writes the histogram data to a CSV file
GCI: 3/4/14 -- (1) Modified the hist_makeData function to include a keyword to
    append the data set name so that it doesn't conflict with previously saved 
    data. Currently this has been added to allow for custom histogram data
    to be
    saved in addition to standard histogram data, however, this will also
    enable
    the funciton to save multiple iterations or versions of the histogram
    for a
    single volume (e.g. for binsize evaluation, or filter evaluation).
    (2) Created a new function specifically to generate histogram data. This 
    function returns three numpy arrays corresponding to the bin voxel count 
    (hist), average intensity value for each bin (bin_avg) and the edge value
    for each bin (bin_edges). The results from this function are then used by 
    the plotting funciton, as well as the two saving functions that are
    included
    for saving the data either to the H5 file, or to a separate CSV file.
GCI: 9/18/14 -- Updating hist functions for incorporation into
VisTrails
"""

import numpy as np


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
    print "initial range equals: " + str(range_initial)
    scale_factor = (range_final)/(range_initial)
    print "scale factor equals"
    print scale_factor
    normalized_data = scale_factor * (src_float - min_initial)
    print "normalized data equals: "
    print normalized_data
    scaled_data = normalized_data + min_initial
    print "scaled data before floor"
    print scaled_data
    if 'int' in out_dType:
        result = np.floor(scaled_data)
    print "scaled data after floor"
    print result
    result = result.astype(out_dType)
    print "result after dType conversion"
    print result
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
    bin_avg = np.empty(len(hist))
    intensity = iter(bin_edges)
    row_count = 0
    right_bin_edge = next(intensity)
    for left_bin_edge in bin_edges:
        right_bin_edge = next(intensity)
        bin_avg[row_count] = (left_bin_edge + right_bin_edge) / 2
        row_count += 1
        if right_bin_edge == bin_edges[len(bin_edges) - 1]:
            break
    return hist, bin_edges, bin_avg


