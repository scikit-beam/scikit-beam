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
