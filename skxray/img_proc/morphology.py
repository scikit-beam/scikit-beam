# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Oct. 2013
"""
This module includes functions focused on grayscale and binary morphological
modification functions. Common ussage of these functions includes noise
removal, material isolation, and region growing.
"""


def extract_material(src_data,
                     material_value):
    """
    This function allows the extraction of all voxels that have the
    defined intensity value. Typically this function will be utilized in order
    to isolate a single material from a segmented data set for additional
    visualization, manipulation or comparison.

    Parameters
    ----------
    src_data : array
        Specifies the first reference data

    material_value : int
        Specifies value of all voxels that you want to extract from the data set
        into a seperate binary volume.

    Returns
    -------
    output : array
        Returns the resulting array to the designated variable
    """
    output = src_data == material_value
    return output


def extract_all_else(src_data,
                     material_value):
    """
    This function allows the extraction of all voxels that do not have the
    defined intensity value. Typically this function will be utilized in order
    to isolate a single material from a segmented data set for additional
    visualization, manipulation or comparison.

    Parameters
    ----------
    src_data : array
        Specifies the first reference data

    material_value : int
        Specifies value of all voxels that you want to exclude from the
        resulting binary volume.

    Returns
    -------
    output : array
        Returns the resulting binary array to the designated variable

    """
    output = src_data != material_value
    return output
