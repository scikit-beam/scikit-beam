# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Oct. 2013
"""
This module contains all of the tools which quantify segmented data sets.
The typical input required for all of the tools is a labeled volume where
each region or material of interest has been segmented and assigned an
individual label (integer value).
"""

import numpy as np
import scipy.ndimage.measurements as measure


results_dict = {}
def create_label_field(binary_vol_list,
                       material_names_list,
                       md_dict):
    #md_dict should be optional input. If present then add to it.
    return result_lblfield, mislabeled_voxel_vol, md_dict

def q_vol_hist(src_labelField,
                num_bins, 
                pd_function=False):
    """
    This function is a trial volume quantification tool which
    generates  a histogram of the labeled volume. Each individual
    material has its  own bin and corresponding voxel count. The
    subsequent Q_VOL function  appears to be more readily useful, but
    I'm retaining this function for  future development as histogram
    analysis and comparison of features or  materials can be useful
    for quantitative comparison of results.


    Parameters
    ----------
    src_labelField : ndarray
         The segmented data set converted to a label field

    num_bins : integer
         number of bins in histogram

    pd_function : bool, optional
        passed to `density` kwarg of `np.histogram`

    Returns
    -------
    output : array
        Currently returns an array containing material number
        (bin_edges) and quantified volume (voxel count).
    """

    hist, bin_edges = np.histogram(src_labelField, 
                                   bins=num_bins,
                                   density=pd_function)
    quant_array = np.array(hist, 
                           bin_edges)
    output = quant_array
    return output


def q_vol(src_labelField, md_dict=None):
    """
    This function quantifies the volume of each material or object
    contained in the parent label field, and returns a dictionary
    containing Material number, phase number, volume and unit attribute
    for the quantified volume.

    Parameters
    ----------
    src_labelField : ndarray
        The segmented data set converted to a label field

    md_dict : dict, optional
        Metadata dictionary associated with the labeled data set.
        If the dictionary is included, then the material names are extracted
        from the dictionary as the results are computed.

    Returns
    -------
    output : dict
        a dictionary containing material number, phase number,
        volume and unit attribute for the quantified volume.
    """
    z_dim, y_dim, x_dim = src_labelField.shape
    binary_vol = np.ones((z_dim, 
                          y_dim, 
                          x_dim))
    num_labels = np.amax(src_labelField)
    material_vols = measure.sum(binary_vol, 
                                src_labelField, 
                                range(num_labels))
    measures = {}
    for x in range(len(material_vols)):
        material_ID = 'Material_' + str(x)
        # size demo is not defined
        measures[material_ID] = {'name': 'Phase_' + str(x),
                                 'volume': {'value': material_vols[x],
                                            'units': 'voxels'}}
        #print measures[material_ID]['name'] + ' Measured Volume'
        #print (str(measures[material_ID]['volume']['value']) + ' ' +
            measures['Material_0']['volume']['units'])
    vol_rec = np.recarray((len(measures.keys())), dtype=[('material ID', object), ('measured volume', float), ('units', object)])
    counter = 0
    for x in measures.keys():
        vol_rec['material ID'][counter] = measures[x]['name']
        vol_rec['measured volume'][counter] = measures[x]['volume']['value']
        vol_rec['units'][counter] = measures[x]['volume']['units']
        counter = counter + 1
    return measures, vol_rec

def save_measure_CSV (write_file_name, data, column_titles):
    np.savetxt(write_file_name, 
               data, 
               delimiter=',',
               header=column_titles)


def surf_area (label_field, label_dict = None):
    """
    label_dict should take the form of:
        label_dict = {"Phase_0_name" : "assigned value as int",
                      "Phase_1_name" : "assigned value as int"
                      }
    """
    if len(label_field.shape) == 2:
        z_dim = 0
        y_dim, x_dim = label_field.shape
    elif len(label_field.shape) == 3:
        z_dim, y_dim, x_dim = label_field.shape
    if label_field.dtype == bool:
    while z in z_dim < (z_dim-1):
        while y in y_dim < (y_dim-1):
            while x in x_dim < (x_dim-1):
