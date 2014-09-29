# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Oct. 2013
"""
This class is includes functions focused on grayscale and binary morphological 
modification functions. Common ussage of these functions includes noise removal, 
material isolation, and region growing.
"""
""" 
 REVISION LOG: (FORMAT: "PROGRAMMER INITIALS: DATE -- RECORD")
 -------------------------------------------------------------
GCI: 2/12/2014 -- (1) Modifying documentation of the package functions for 
    inclusion in the bulk module pull to GITHUB
    (2) Added the binary propagation function (propagate_binary) to the list of 
    available functions.
GCI: 2/20/2014 -- Updated documentation to docstring format
    Changed file name from C7_morphology.py to morphology.py
    Changed structure from class to simple module of function definitions
GCI: 3/5/2014 -- Removed the importation of the display module. Tool revisions
    have removed all dependencies on this module, at least for the time being.
    Changed function calls to be more readable by separating all variable names
    and calls.
"""

from scipy.ndimage.morphology import binary_opening
from scipy.ndimage.morphology import binary_closing
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.morphology import grey_opening
from scipy.ndimage.morphology import grey_closing
from scipy.ndimage.morphology import grey_erosion
from scipy.ndimage.morphology import grey_dilation
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.morphology import binary_propagation
