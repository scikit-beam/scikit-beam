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

binary_opening.__doc__ = (
    """
    This function executes a morphological opening operation on binary data 
    sets. Morphological opening couples erosion (first operation) and dilation 
    (second operation). Opening can be used to remove small objects from the 
    foreground.
    
    Parameters
    ----------
    src_data : ndarray
        Identify source data for modification

    struct_size : tuple
        Define the 2D or 3D structure size for the erosion and dilation 
        operations. Note that for a cube-type structuring element containing 
        all ones, only the directly connected voxels will be altered. 
        Diagonally connected voxels will remain unchanged unless they become 
        directly connected during an iterative erosion or dilation operation.

    Returns
    -------
    output : ndarray
        Return operation result to specified variable
    """
)


binary_closing.__doc__ = (
    """
    This function executes a morphological closing operation on binary data 
    sets. Morphological closing couples dilation (first operation) and erosion 
    (second operation). Closing can be used to remove small holes from the 
    foreground.
    
    Parameters
    ----------
    src_data : ndarray
        Identify source data for modification
    
    struct_size : tuple
        Define the 2D or 3D structure size for the erosion and dilation 
        operations. Note that for a cube-type structuring element containing 
        all ones, only the directly connected voxels will be altered. 
        Diagonally connected voxels will remain unchanged unless they become 
        directly connected during an iterative erosion or dilation operation.
    
    Returns
    -------
    output : ndarray
        Return operation result to specified variable
    """
)


binary_erosion.__doc__ = (
    """
    This function executes a morphological erosion operation on binary data 
    sets. Morphological erosion removes the outer layer of voxels from 
    foreground objects based on the structuring element defined as an input 
    parameter. Erosion can be used to remove small objects as well as to aid in 
    evaluating material connectivity or source/seed regions of a material.

    Parameters
    ----------
    src_data : ndarray
        Identify source data for modification
    
    struct_size: tuple
        Define the 2D or 3D structure size for the operation. Note that for a 
    cube-type structuring element containing all ones, only the directly 
    connected voxels will be altered. Diagonally connected voxels will remain 
    unchanged unless they become directly connected during an iterative erosion 
    or dilation operation.
    
    Returns
    ----------
    output : ndarray
        Return operation result to specified variable
    """
)


binary_dilation.__doc__ = (
    """
    This function executes a morphological dilation operation on binary data 
    sets. Morphological dilation expands the outer layer of voxels in 
    foreground objects based on the structuring element defined as an input 
    parameter. Dilation can be used to remove holes from foreground objects. 
    The operation can also be used in evaluating material connectivity.
    
    Parameters
    ----------
    src_data : ndarray
        Identify source data for modification
    
    struct_size : tuple
        Define the 2D or 3D structure size for the operation. Note that for a 
        cube-type structuring element containing all ones, only the directly
        connected voxels will be altered. Diagonally connected voxels will
        remain unchanged unless they become directly connected during an
        iterative erosion or dilation operation.
    
    Returns
    -------
    output : ndarray
        Return operation result to specified variable
    """
)


grey_erosion.__doc__ = (
    """
    This function executes a morphological erosion operation on grayscale data 
    sets. Morphological erosion removes the outer layer of voxels from 
    foreground objects based on the structuring element defined as an input 
    parameter. Erosion can be used to remove small objects as well as to aid in 
    evaluating material connectivity or source/seed regions of a material. In 
    grayscale datasets this operation is essentially a minimum filter executed 
    within a moving window, defined by the structuring element input.
    
    Parameters
    ----------
    src_data : ndarray
        Identify source data for modification
    
    struct_size : tuple
        Define the 2D or 3D structure size for the operation. Note that for a 
        cube-type structuring element containing all ones, only the directly 
        connected voxels will be altered. Diagonally connected voxels will 
        remain unchanged unless they become directly connected during an 
        iterative erosion or dilation operation.
    
    Returns
    -------
    output : ndarray
        Return operation result to specified variable
    """
)


grey_dilation.__doc__ = (
    """
    This function executes a morphological dilation operation on grayscale data 
    sets. Morphological dilation expands the outer layer of voxels in 
    foreground objects based on the structuring element defined as an input 
    parameter. Dilation can be used to remove holes from foreground objects. 
    The operation can also be used in evaluating material connectivity. In 
    grayscale datasets this operation is essentially a maximum filter executed 
    within a moving window, defined by the structuring element input.

    Parameters
    ----------
    src_data : ndarray
        Identify source data for modification
    
    struct_size : tuple
        Define the 2D or 3D structure size for the operation. Note that for a 
        cube-type structuring element containing all ones, only the directly 
        connected voxels will be altered. Diagonally connected voxels will 
        remain unchanged unless they become directly connected during an 
        iterative erosion or dilation operation.

    Returns
    -------
    output : ndarray
        Return operation result to specified variable
    """
)


grey_opening.__doc__ = (
    """
    This function executes a morphological opening operation on grayscale data 
    sets. Morphological opening couples erosion (first operation) and dilation 
    (second operation). Opening can be used to remove small objects from the 
    foreground. In grayscale datasets this operation is essentially a coupling 
    of a minimum and maximum filter executed within a moving window, defined by 
    the structuring element input.

    Parameters
    ----------
    src_data : ndarray
        Identify source data for modification
    
    struct_size : tuple
        Define the 2D or 3D structure size for the erosion and dilation 
        operations. Note that for a cube-type structuring element containing 
        all ones, only the directly connected voxels will be altered. 
        Diagonally connected voxels will remain unchanged unless they become 
        directly connected during an iterative erosion or dilation operation.
    
    Returns
    -------
    output : ndarray
        Return operation result to specified variable
    """
)


grey_closing.__doc__ = (
    """
    This function executes a morphological closing operation on grayscale data 
    sets. Morphological closing couples dilation (first operation) and erosion 
    (second operation). Closing can be used to remove small holes from the 
    foreground. In grayscale datasets this operation is essentially a coupling 
    of a maximum and minimum filter executed within a moving window, defined by 
    the structuring element input.
    
    Parameters
    ----------
    src_data : ndarray
        Identify source data for modification
    
    struct_size : tuple
        Define the 2D or 3D structure size for the erosion and dilation 
        operations. Note that for a cube-type structuring element containing 
        all ones, only the directly connected voxels will be altered. 
        Diagonally connected voxels will remain unchanged unless they become 
        directly connected during an iterative erosion or dilation operation.
    
    Returns
    -------
    output : ndarray
        Return operation result to specified variable
    """
)


binary_fill_holes.__doc__ = (
    """
    This function executes a morphological opening operation on binary data 
    sets. Morphological opening couples erosion (first operation) and dilation 
    (second operation). Opening can be used to remove small objects from the 
    foreground.
    
    Parameters
    ----------
    src_data : ndarray
        Identify source data for modification
    
    struct_size : tuple
        Define the 2D or 3D structure size for the erosion and dilation 
        operations. Note that for a cube-type structuring element containing 
        all ones, only the directly connected voxels will be altered. 
        Diagonally connected voxels will remain unchanged unless they become 
        directly connected during an iterative erosion/dilation.
    
    Returns
    -------
    output : ndarray
        Return operation result to specified variable
    """
)


binary_propagation.__doc__ = (
    """
    This function allows for dilation of foreground objects within a binary data 
    set upto boundaries defined using the mask_region operator. Morphological 
    propagation, coupled with iterative erosion, can be used to remove holes 
    from foreground objects while retaining the boundaries of larger, bulk 
    foreground objects. Propagation occurs until no changes occur on the next 
    successive dilation.
    
    Parameters
    ----------
    src_data : ndarray
        Identify source data for modification
    
    struct_size : tuple
        Define the 2D or 3D structure size for the erosion and dilation 
        operations. Note that for a cube-type structuring element containing 
        all ones, only the directly connected voxels will be altered. 
        Diagonally connected voxels will remain unchanged unless they become 
        directly connected during an iterative erosion/dilation.
        
    mask_region : ndarray
        Binary mask array which defines the regions into which regions can grow 
        during propagation.
    
    point_origin : tuple
        Defines the location where the operation should begin within the source 
        array.
    
    Returns
    -------
    output : ndarray
        Return operation result to specified variable
    """
)
