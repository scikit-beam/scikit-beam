from __future__ import absolute_import, division, print_function

import logging
import os

import numpy as np


def _read_amira(src_file):
    """
    Reads all information contained within standard AmiraMesh data sets.
    Separate the header information from the image/volume, data.

    Parameters
    ----------
    src_file : str
        The path and file name pointing to the AmiraMesh file to be loaded.

    Returns
    -------
    am_header : list of strings
        This list contains all of the raw information contained in the
        AmiraMesh file header. Contains all of the raw header information
    am_data : str
        A compiled string containing all of the image array data, that was
        stored in the source AmiraMesh data file. Contains the raw image data
    """
    am_header = []
    am_data = []
    with open(os.path.normpath(src_file), "r") as input_file:
        while True:
            line = input_file.readline()
            am_header.append(line)
            if line == "# Data section follows\n":
                input_file.readline()
                break
        am_data = input_file.read()
    return am_header, am_data


def _amira_data_to_numpy(am_data, header_dict, flip_z=True):
    """
    Transform output of `_read_amira` to a numpy array of the dtype listed in
    the AmiraMesh header dictionary.  The standard format for Avizo Binary
    files is IEEE binary. Big or little endian-ness is stipulated in the header
    information, and is be assessed and taken into account by this function as
    well, during the conversion process.

    Parameters
    ----------
    am_data : str
        String object containing all of the image array data, formatted as IEEE
        binary. Current dType options include:
            float
            short
            ushort
            byte

    header_dict : dict
        Metadata dictionary containing all relevant attributes pertaining to
        the image array. This metadata dictionary is the output from the
        function `_create_md_dict`.

    flip_z : bool, optional.
        Defaults to True
        This option is included because the .am data sets evaluated thus far
        have opposite z-axis indexing than numpy arrays. This switch currently
        defaults to "True" in order to ensure that z-axis indexing remains
        consistent with data processed using Avizo.
        Setting this switch to "True" will flip the z-axis during processing,
        and a value of "False" will keep the array is initially assigned during
        the array reshaping step.

    Returns
    -------
    output : ndarray
        Numpy ndarray containing the image data converted from the AmiraMesh
        file. This data array is ready for further processing using the NSLS-II
        function library, or other operations able to operate on numpy arrays.
    """
    Zdim = header_dict["array_dimensions"]["z_dimension"]
    Ydim = header_dict["array_dimensions"]["y_dimension"]
    Xdim = header_dict["array_dimensions"]["x_dimension"]
    # Strip out null characters from the string of binary values
    # Dictionary of the encoding types for AmiraMesh files
    am_format_dict = {"BINARY-LITTLE-ENDIAN": "<", "BINARY": ">", "ASCII": "unknown"}
    # Dictionary of the data types encountered so far in AmiraMesh files
    am_dtype_dict = {"float": "f4", "short": "h4", "ushort": "H4", "byte": "b"}
    # Had to split out the stripping of new line characters and conversion
    # of the original string data based on whether source data is BINARY
    # format or ASCII format. These format types require different stripping
    # tools and different string conversion tools.
    if header_dict["data_format"] == "BINARY-LITTLE-ENDIAN":
        data_strip = am_data.strip("\n")
        flt_values = np.fromstring(
            data_strip, (am_format_dict[header_dict["data_format"]] + am_dtype_dict[header_dict["data_type"]])
        )
    if header_dict["data_format"] == "ASCII":
        data_strip = am_data.translate(None, "\n")
        string_list = data_strip.split(" ")
        string_list = string_list[0 : (len(string_list) - 2)]
        flt_values = np.array(string_list).astype(am_dtype_dict[header_dict["data_type"]])
    # Resize the 1D array to the correct ndarray dimensions
    # Note that resize is in-place whereas reshape is not
    flt_values.resize(Zdim, Ydim, Xdim)
    output = flt_values
    if flip_z:
        output = flt_values[::-1, ..., ...]
    return output


def _clean_amira_header(header_list):
    """
    Strip the string list of all "empty" characters,including new line
    characters ('\n') and empty lines. Splits each header line (which
    originally is stored as a single string) into individual words, numbers or
    characters, using spaces between words as the separating operator. The
    output of this function is used to generate the metadata dictionary for
    the image data set.

    Parameters
    ----------
    header_list : list of strings
        This is the header output from the function _read_amira()

    Returns
    -------
    clean_header : list of strings
        This header list has been stripped and sorted and is now ready for
        populating the metadata dictionary for the image data set.
    """
    clean_header = []
    for row in header_list:
        split_header = filter(None, [word.translate(None, ',"') for word in row.strip("\n").split()])
        clean_header.append(split_header)
    return clean_header


def _create_md_dict(clean_header):
    """
    Populates the a dictionary with all information pertinent to the image
    data set that was originally stored in the AmiraMesh file.

    Parameters
    ----------
    clean_header : list of strings
        This is the output from the _sort_amira_header function.

    """
    # Avizo specific metadata
    md_dict = {
        "software_src": clean_header[0][1],
        "data_format": clean_header[0][2],
        "data_format_version": clean_header[0][3],
    }
    if md_dict["data_format"] == "3D":
        md_dict["data_format"] = clean_header[0][3]
        md_dict["data_format_version"] = clean_header[0][4]

    for header_line in clean_header:
        hl = header_line
        if "define" in hl:
            hl = hl
            md_dict["array_dimensions"] = {
                "x_dimension": int(hl[hl.index("define") + 2]),
                "y_dimension": int(hl[hl.index("define") + 3]),
                "z_dimension": int(hl[hl.index("define") + 4]),
            }
        elif "Content" in hl:
            md_dict["data_type"] = hl[hl.index("Content") + 2]
        elif "CoordType" in hl:
            md_dict["coord_type"] = hl[hl.index("CoordType") + 1]
        elif "BoundingBox" in hl:
            hl = hl
            md_dict["bounding_box"] = {
                "x_min": float(hl[hl.index("BoundingBox") + 1]),
                "x_max": float(hl[hl.index("BoundingBox") + 2]),
                "y_min": float(hl[hl.index("BoundingBox") + 3]),
                "y_max": float(hl[hl.index("BoundingBox") + 4]),
                "z_min": float(hl[hl.index("BoundingBox") + 5]),
                "z_max": float(hl[hl.index("BoundingBox") + 6]),
            }

            # Parameter definition for voxel resolution calculations
            bbox = [
                md_dict["bounding_box"]["x_min"],
                md_dict["bounding_box"]["x_max"],
                md_dict["bounding_box"]["y_min"],
                md_dict["bounding_box"]["y_max"],
                md_dict["bounding_box"]["z_min"],
                md_dict["bounding_box"]["z_max"],
            ]
            dims = [
                md_dict["array_dimensions"]["x_dimension"],
                md_dict["array_dimensions"]["y_dimension"],
                md_dict["array_dimensions"]["z_dimension"],
            ]

            # Voxel resolution calculation
            resolution_list = []
            for index in np.arange(len(dims)):
                if dims[index] > 1:
                    resolution_list.append((bbox[(2 * index + 1)] - bbox[(2 * index)]) / (dims[index] - 1))
                else:
                    resolution_list.append(0)
            # isotropy determination (isotropic res, or anisotropic res)
            if (
                resolution_list[1] / resolution_list[0] > 0.99
                and resolution_list[2] / resolution_list[0] > 0.99
                and resolution_list[1] / resolution_list[0] < 1.01
                and resolution_list[2] / resolution_list[0] < 1.01
            ):
                md_dict["resolution"] = {"zyx_value": resolution_list[0], "type": "isotropic"}
            else:
                md_dict["resolution"] = {
                    "zyx_value": (resolution_list[2], resolution_list[1], resolution_list[0]),
                    "type": "anisotropic",
                }

        elif "Units" in hl:
            try:
                units = str(hl[hl.index("Units") + 2])
                md_dict["units"] = units
            except Exception:
                logging.debug(
                    "Units value undefined in source data set. " "Reverting to default units value of pixels"
                )
                md_dict["units"] = "pixels"
        elif "Coordinates" in hl:
            coords = str(hl[hl.index("Coordinates") + 1])
            md_dict["coordinates"] = coords
    return md_dict


def load_amiramesh(file_path):
    """
    Load and convert an AmiraMesh binary file to a numpy array.

    Parameters
    ----------
    file_path : str
        The path and file name of the AmiraMesh file to be loaded.

    Returns
    -------
    md_dict : dict
        Dictionary containing all pertinent header information associated with
        the data set.
    np_array : ndarray
        An ndarray containing the image data set to be loaded. Values contained
        in the resulting volume are set to be of float data type by default.
    """
    header, data = _read_amira(file_path)
    clean_header = _clean_amira_header(header)
    md_dict = _create_md_dict(clean_header)
    np_array = _amira_data_to_numpy(data, md_dict)
    return md_dict, np_array
