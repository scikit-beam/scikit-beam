# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Oct. 2013
"""
This module includes functions focused on grayscale and binary morphological
modification functions. Common ussage of these functions includes noise
removal, material isolation, and region growing.

Some tools are being added directly from scipy.ndimage for use in the skxray
tool set that is included for image processing in VisTrails. These tools are
wrappoed into the complete toolkit through the API. The tools from
scipy.ndimage include:
    binary_opening,
    binary_closing,
    binary_erosion,
    binary_dilation,
    grey_opening,
    grey_closing,
    grey_erosion,
    grey_dilation,
    binary_fill_holes,
    binary_propagation
"""
