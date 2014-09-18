# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Oct. 2013
"""
This class is designed to evaluate data set histograms by providing
functions for easily plotting, evaluating, saving, and modifiying histograms.
"""
"""
REVISION LOG: (FORMAT: "PROGRAMMER INITIALS: DATE -- RECORD")
GCI: 9/18/14 -- Updating hist functions for incorporation into VisTrails cut
    the histogram IO functions from histops.py for inclusion in the IO file
    options.
    We may want to consider making these functions more general so that we have
    a "save_to_CSV" and "save_to_H5" function set that can be applied
    generally.
"""

import numpy as np
import h5py
import csv
import time


def hist_save_H5(hist,
                 bin_edges,
                 bin_avg,
                 src_file,
                 src_dataset_name,
                 dSet_append=''):
    """
    This function evaluates the histogram of the source data set and returns
    the result as a zipped object containing two lists corresponding to "bin"
    values and "count" values.

    Parameters
    ----------
    hist : ndarray
        1xN numpy array containing all of the bin count values

    bin_edges : ndarray
        1xN numpy array containing the intensity values of the bin edges.

    bin_avg : ndarray
        1xN numpy array containing the average intensity values of the bins.

    src_file : h5
        Open hdf5 file
        Specifies the source file containing the data set from which you want
        to measure the histogram.

    src_dataset_name : str
        hdf5 data set key name
        Specifies the specific data set from which the histogram is to be
        generated.

    dSet_append : str
        Optional string entry for differentiating histogram data generated from
        the same data set.
            (e.g. dSet_append = '_01' would result in the histogram being saved
            to the h5 data set 'src_data_name_01')

    Returns
    -------
    output : h5
        Two 1xN data set arrays appended to existing h5 file
        Function returns two 1xN data set array objects
            (1) array containing the quantified values for each bin (hist)
            (2) array containing the mean intensity value for each bin
                (bin_edges)
        The two arrays are contained within a GROUP named after the source data
        set
    """
    bin_size = (bin_edges[1] - bin_edges[0])
    num_bins = len(hist)
    voxel_units = src_file["exchange"]["voxel_size"].attrs["Units"]
    grp_measure = src_file["measurements"]
    src_dataset_name += dSet_append
    if 'histogram' in grp_measure:
        grp_measure["histogram"]
    else:
        grp_measure.create_group("histogram")
    if src_dataset_name in grp_measure["histogram"]:
        grp_measure["histogram"][src_dataset_name]
    else:
        grp_measure["histogram"].create_group(src_dataset_name)

    hist_grp = grp_measure["histogram"][src_dataset_name]
    if 'hist' in hist_grp:
        raise ValueError("Histogram Data already exists.")
    else:
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
    if np.amax(hist) <= 1:
        hist_grp["hist"].attrs.create("Units",
                                      "Normalized Probability Density")
    else:
        hist_grp["hist"].attrs.create("Units",
                                      "Voxel Count")


def hist_save_CSV(write_file_name,
                  hist,
                  bin_avg):
    """
    This function saves histogram data as a CSV file saved to the file name
    specified as an input.

    Parameters
    ----------
    write_file_name : str
        String containing the full path and file name to which the resulting
        file will be saved.

    hist : array
        1xN numpy array containing all of the bin count values

    bin_avg : array
        1xN numpy array containing the average intensity values of the bins.


    Returns
    -------
    output : CSV text file (2xN)
        Function returns a 2xN comma separated text file where the first column
        contains the average bin intensity value, and the second column
        contains
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


def hist_load(src_file, file_type):
    """
    This function loads histogram data previously saved to either an hdf5 file
    or csv file.

    Parameters
    ----------
    src_file : str
        variable link to open hdf5 or csv file
        This keyword can either specify an open hdf5 file containing previously
        saved histogram data, or the full path and filename of a csv file
        containing previously saved histogram data.

    file_type : str
        Options include:
            "HDF5"
            "CSV"

    Returns
    -------
     hist : numpy array
        1xN numpy array containing all of the bin count values

    bin_edges : numpy array
        1xN numpy array containing the intensity values of the bin edges.

    bin_avg : numpy array
        1xN numpy array containing the average intensity values of the bins.
    """


    # TODO Finish writing this function after other tools have been updated
    # this function should enable loading from CSV files and from HDF5,
    # or there should be two separate functions. One for HDF5 and one for
    # CSV...
    pass
