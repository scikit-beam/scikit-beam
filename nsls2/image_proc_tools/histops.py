# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Oct. 2013
"""
This class is designed to evaluate data set histograms by providing 
functions for easily plotting, evaluating, saving, and modifiying histograms.
"""
"""
REVISION LOG: (FORMAT: "PROGRAMMER INITIALS: DATE -- RECORD")
GCI: 2/11/2014 -- Modifying documentation of the package functions for 
    inclusion in the bulk module pull to GITHUB
GCI: 2/19/2014 -- Updating module to docstring format.
    Changed filename from C3_histops.py to histops.py
    TODO Finish Adding function for saving histogram data as a CSV file or HDF5 
    data set.
    TODO Finish adding function for loading previously saved histogram data.
GCI: 2/28/2014 -- Added the functions: 
    hist_makeData: which writes histogram data to the HDF5 file
    hist_makeCSV: which writes the histogram data to a CSV file
GCI: 3/4/14 -- (1) Modified the hist_makeData function to include a keyword to 
    append the data set name so that it doesn't conflict with previously saved 
    data. Currently this has been added to allow for custom histogram data to be
    saved in addition to standard histogram data, however, this will also enable
    the funciton to save multiple iterations or versions of the histogram for a 
    single volume (e.g. for binsize evaluation, or filter evaluation).
    (2) Created a new function specifically to generate histogram data. This 
    function returns three numpy arrays corresponding to the bin voxel count 
    (hist), average intensity value for each bin (bin_avg) and the edge value 
    for each bin (bin_edges). The results from this function are then used by 
    the plotting funciton, as well as the two saving functions that are included
    for saving the data either to the H5 file, or to a separate CSV file.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import csv
import time


def rescale_intensity_values(src_data, 
                             new_max = 254, 
                             new_min = 0, 
                             out_dType = 'uint8'):
    """
    The purpose of this function is to allow easy conversion, scaling, or 
    expansion of data set intensity ranges for additional histogram analysis or 
    data manipulation.
    
    Parameters
    ----------
    src_data : numpy array 
        Specifies the data set you want to rescale
    
    new_max : scalar value of source dtype
        Specify the new maximum value for the data set. Default 
        value is 254

    new_min : scalar value of source dtype
        Specify the new minimum value for the data set. Default 
        value is 0

    out_dType : string
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
    output : NxN or NxNxN numpy array
        Returns the resulting array to the designated variable

    Example
    -------
    Given an 8-bit binary or trinary volume that has been set to have a range of 
    0 and 255 (for a binary volume) or 85, 170 and 255 (for a trinary volume) 
    and reset the data intensity range to a fixed range of 0 and 1 (or 1 and 2) 
    for a binary data set, or 0, 1, 2 (1, 2, 3) for a trinary data set.
 
    """
    src_float = np.float32(src_data)
    max_value = np.amax(src_float)
    min_value = np.amin(src_float)
    if min_value < 0:
        normalizing_const = max_value - (min_value) + 1
    else:
        normalizing_const = max_value
    normalized_data = src_float / normalizing_const
    if np.amin(normalized_data) != 0:
        normal_shift = np.amin(normalized_data)
        normalized_data = normalized_data - normal_shift
    scale_factor = new_max - new_min + 1
    result = normalized_data * scale_factor
    result = result + new_min
    result = np.around(result)
    result = result.astype(out_dType)
    print 'The source volume has been rescaled.'
    return result

def hist_make (src_data,
               num_bins,
               pd_function):
    """
    This function evaluates the histogram of the source data set and plots the 
    result using a standard format.

    Parameters
    ----------
    src_data : numpy array or HDF5 data set
        Specifies the source data set from which you want to measure the 
        histogram.

    num_bins : int
        Specify the number of bins to include in the histogram as an integer.
        
    pd_function : bool
        Identify whether the histogram data should be normalized as a 
        probability density histogram or not.
        Options:
            True -- Histogram data is normalized to range from 0 to 1
            False -- Histogram data reported simply as "counts" (e.g. Voxel 
                     Count)

    Returns
    -------
    output : 3 - 1xN numpy arrays
        (1) hist -- contains all of the actual bin measurements (e.g. voxel 
            counts)
        (2) bin_avg -- contains the average intensity value for each bin.
            NOTE: the length of this array is equal to the length of the hist 
            array
        (3) bin_edges -- contains the edge values for each bin
            NOTE: the length of this array is 1 larger than the length of the 
            hist array (e.g. len(bin_edges) = len(hist) + 1)
    """
    hist, bin_edges = np.histogram(src_data, 
                                   bins=num_bins, 
                                   density=pd_function)
    bin_avg = np.empty(len(hist))
    I = iter(bin_edges)
    row_count = 0
    bin_edge_A = next(I)
    for x in bin_edges:
        bin_edge_B = next(I)
        bin_avg[row_count] = (bin_edge_A + bin_edge_B)/2
        row_count = row_count + 1
        bin_edge_A = bin_edge_B
        if bin_edge_B == bin_edges[len(bin_edges) - 1]: break
    return hist, bin_edges, bin_avg


def hist_plot (hist,
               bin_edges,
               plot_min,
               plot_max,
               yscale,
               show_plot=None):
    """
    This function evaluates the histogram of the source data set and plots the 
    result using a standard format.

    Parameters
    ----------
    hist : numpy array
        1xN numpy array containing all of the bin count values

    bin_edges : numpy array
        1xN numpy array containing the intensity values of the bin edges.
        
    plot_min : same dtype as bin_edges numpy array
        Specify the minimum range value for the histogram plot
    
    plot_max : same dtype as bin_edges numpy array
        Specify the maximum range value ofr the histogram plot
    
    y_scale : string
        Specify the desired Y-axis scale. Available options are:
            'linear' -- for an arithmetic scale
            'log' -- for a logarithmic scale

    Returns
    -------
    output : 2-D histogram plot
        2-D plot of the data set histogram, plotted using matplotlib.pyplot
    """
    plt.bar(bin_edges[:-1], 
            hist, 
            width=1, 
            align='center')
    plt.ylabel('Voxel Count', 
               weight='heavy')
    if np.amax(hist) <= 1:
        plt.xlabel('Probability Density', 
                   weight='heavy')
        plt.title('Probability Density Histogram')
    elif np.amax(hist) > 1:
        plt.xlabel('Voxel Intensity', 
                   weight='heavy')
        plt.title('Volume Histogram')
    plt.yscale(yscale)
    plt.xlim(plot_min, 
             plot_max)
    if show_plot == None:
        plt.show()


def hist_save_H5 (hist,
                  bin_edges,
                  bin_avg,
                  src_file,
                  src_dataset_name, 
                  dSet_append=''):
    """
    This function evaluates the histogram of the source data set and returns the 
    result as a zipped object containing two lists corresponding to "bin" values 
    and "count" values.

    Parameters
    ----------
    hist : numpy array
        1xN numpy array containing all of the bin count values

    bin_edges : numpy array
        1xN numpy array containing the intensity values of the bin edges.
        
    bin_avg : numpy array
        1xN numpy array containing the average intensity values of the bins.
    
    src_file : h5 data set
        Specifies the source file containing the data set from which you want to
        measure the histogram.
    
    src_dataset_name : h5 key name
        Specifies the specific data set from which the histogram is to be 
        generated.
    
    dSet_append : string
        Optional string entry for differentiating histogram data generated from 
        the same data set.
            (e.g. dSet_append = '_01' would result in the histogram being saved 
            to the h5 data set 'src_data_name_01')

    Returns
    -------
    output : Two 1xN h% data set arrays
        Function returns two 1xN data set array objects 
            (1) array containing the quantified values for each bin (hist)
            (2) array containing the mean intensity value for each bin 
                (bin_edges)
        The two arrays are contained within a GROUP named after the source data 
        set
    """
    bin_size = (bin_edges[1]-bin_edges[0])
    num_bins = len(hist)
    voxel_units = src_file["exchange"]["voxel_size"].attrs["Units"]
    grp_measure = src_file["measurements"]
    src_dataset_name = src_dataset_name + dSet_append
    try:
        grp_measure["histogram"]
    except:
        grp_measure.create_group("histogram")
    try:
        grp_measure["histogram"][src_dataset_name]
    except:
        grp_measure["histogram"].create_group(src_dataset_name)
    hist_grp = grp_measure["histogram"][src_dataset_name]
    try:
        hist_grp["hist"]
    except:
        hist_grp.create_dataset("hist", 
                                data=hist, 
                                compression='gzip')
        hist_grp.create_dataset("bins_avg", 
                                data=bin_avg, 
                                compression='gzip')
        hist_grp.create_dataset("bin_edges", 
                                data=bin_edges, 
                                compression='gzip') 
        hist_grp.attrs.create("Creation Time", 
                              time.ctime())
        hist_grp.attrs.create("bin count", 
                              num_bins)
        hist_grp["bins_avg"].attrs.create("bin size", 
                                      bin_size)
        hist_grp["bins_avg"].attrs.create("Units", 
                                      voxel_units)
        hist_grp["bin_edges"].attrs.create("bin size", 
                                      bin_size)
        hist_grp["bin_edges"].attrs.create("Units", 
                                      voxel_units)
    else:
        raise ValueError ("Histogram Data already exists.")
    if np.amax(hist) <= 1:
        hist_grp["hist"].attrs.create("Units", 
                                      "Normalized Probability Density")
    else:
        hist_grp["hist"].attrs.create("Units", 
                                      "Voxel Count")


def hist_save_CSV (write_file_name, 
                   hist, 
                   bin_avg):
    """
    This function evaluates the histogram of the source data set and saves the 
    resulting histogram data as a CSV file saved to the file name specified as 
    an input.

    Parameters
    ----------
    write_file_name : string
        String containing the full path and file name to which the resulting 
        file will be saved.
    
    hist : numpy array
        1xN numpy array containing all of the bin count values

    bin_avg : numpy array
        1xN numpy array containing the average intensity values of the bins.
    

    Returns
    -------
    output : CSV text file (2xN)
        Function returns a 2xN comma separated text file where the first column 
        contains the average bin intensity value, and the second column contains
        the voxel count information. 
    """
    temp_hist_combine = np.vstack((bin_avg, hist))
    hist_combine = np.transpose(temp_hist_combine)
    if np.amax(hist) <= 1:
        column_titles = "Average Bin Intensity, Normalized Probability Density"
    else:
        column_titles = "Average Bin Intensity, Voxel Count"
    np.savetxt(write_file_name, 
               hist_combine, 
               delimiter=',',
               header=column_titles)


#def hist_load (src_file, file_type):
    """
    This function loads histogram data previously saved to either an hdf5 file 
    or csv file.
    
    Parameters
    ----------
    src_file : variable link to open hdf5 or csv file
        This keyword can either specify an open hdf5 file containing previously 
        saved histogram data, or the full path and filename of a csv file 
        containing previously saved histogram data.
    
    file_type : string
        Options include:
            "HDF5"
            "CSV"

    Returns
    -------
    output : either a new data set in the prescribed HDF5 file, or csv file_type
    """
    #TODO Finish writing this function after other tools have been updated and 
    #added to pyLight github

