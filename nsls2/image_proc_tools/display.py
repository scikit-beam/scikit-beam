# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Oct. 2013
"""
This class provides a suite of functions for displaying and comparing data 
sets. Current implementation utilizes matplotlib.pyplot to display discrete 
cros-sectional slices of selected data sets. However, the next step will 
incorporate visualization tools from scikit-image. Specifically, 
skimage.Viewer.CollectionViewer() will be added to the list of display tools 
to facilitate "scannable" volume display where axis orientations and slice 
selection can be changed using slider bars instead of requiring alteration 
of parameters and successive executions of display funcitons.
"""
"""
REVISION LOG: (FORMAT: "PROGRAMMER INITIALS: DATE -- RECORD")
GCI: 1/9/2014 -- I've been playing with both the skimage-collectionViewer and 
		 mayavi.mlab Collection viewer will allow the searching and 
		 display of our volumetric data sets Mayavi allows for 
		 visulization of both Isosurfaces AND volumes. 
		 NEED TO LOOK INTO 
		 mlab.pipeline.volume
		 mlab.pipeline.scalar_field
		 etc....
GCI: 2/12/2014 -- Modifying documentation of the package functions for 
		   inclusion in the bulk module pull to GITHUB.
GCI: 2/20/2014 -- Modified documentation to docstring format
        changed file name from C8_display.py to display.py
        Changed structure from class to simple module of function defs.
        Added function for using CollectionViewer to view entire volumes using 
        a scrollable set of cross-sections.
"""


import numpy as np
import matplotlib.pyplot as plt
import skimage.viewer.viewers as viewer
import transform as xform


def img_display_basic (src_data, 
                       scheme, 
                       save_file, 
                       file_name):
    """
    This function generates a standardized depiction of the identified 
    cross-section (orthoslice) of the source data set. The ability to save the 
    depicted orthoslice(s) has been incorporated into the function using a 
    keyword.

    Parameters
    ----------
    src_data : NxN or NxNxN numpy array
        Array containinge the data to be displayed. Currently, the executable 
        script the runs the function defines the 2-D cross section to be 
        displayed, however, if a 3-D array is used as input the function 
        automatically displays the XY cross-section in the middle of the volume 
        (z-dim/2).
    
    scheme : string
        Select the color scheme to be displayed
    
    save_file : string
        Identify whether to save the resulting figure or not
        Options:
            "YES"
            "NO"
    
    file_name : string
        String containing the file name under which to save the resulting 
        figure.
    
    Returns
    output : 2D plot of delected cross-section
        Generates and displays the 2-D cross-section
        Optional:
            Save the resulting figure using the specified file name
"""
    data_min = np.amin(src_data)
    data_max = np.amax(src_data)
    plot_min = data_min
    plot_max = data_max
    if len(src_data.shape) == 3:
        z_dim, y_dim, x_dim = src_data.shape
        slice_selection = src_data[z_dim/2,:,:]
    elif len(src_data.shape) == 2:
        y_dim, x_dim = src_data.shape
        slice_selection = src_data[:,:]
    plt.imshow(slice_selection, vmin=plot_min, vmax=plot_max, cmap=scheme)
    plt.title('Sample slice of filtered volume')
    if save_file == "YES":
        plt.savefig(file_name, dpi=600)
    elif save_file == "NO":
        print ("Generated figure not saved. pyPlot operation number: " + 
            file_name)
    plt.show()

def img_display_compare_upDown (src_data1, 
                                src_data2, 
                                scheme1, 
                                scheme2, 
                                save_file, 
                                file_name):
    """
    This function generates a figure consisting of two cross-sectional slices. 
    This can be used to directly compare two volumes, or to evaluate two 
    cross-sections of the same volume. The resulting figure displays the two 
    images in a side-by-side orientation.
    
    Parameters
    ----------
    src_data1 : NxN numpy array
        Array containinge the data to be displayed. 
    
    src_data2 : NxN numpy array
        Array containinge the data to be displayed.  
    
    scheme1 : string
        Select the color scheme with which to display src_data1.

    scheme2 : string
        Select the color scheme with which to display src_data2.
    
    save_file : string
        Identify whether to save the resulting figure or not
        Options:
            "YES"
            "NO"
    
    file_name : string
        String containing the file name under which to save the resulting 
        figure.
    
    Returns
    -------
    output : Two 2D plots of selected cross-sections
        Generates and displays the 2-D cross-sections, one above the other
        Optional:
            Save the resulting figure using the specified file name
"""
    plt.figure()
    plt.subplot(121)
    plt.imshow(src_data1, cmap=scheme1)
    plt.title('Source #1 cross-section', weight= 'heavy')
    plt.subplot(122)
    plt.imshow(src_data2, cmap=scheme2)
    plt.title('Source #2 cross-section', weight='heavy')
    if  save_file == "YES":
        plt.savefig(file_name, dpi=600)
    plt.show()

def img_display_compare_leftRight (src_data1, 
                                   src_data2, 
                                   scheme1, 
                                   scheme2, 
                                   save_file, 
                                   file_name):
    """
    This function generates a figure consisting of two cross-sectional slices. 
    This can be used to directly compare two volumes, or to evaluate two 
    cross-sections of the same volume. The resulting figure displays the two 
    images in a verticle orientation.
    
    Parameters
    ----------
    src_data1 : NxN numpy array
        Array containinge the data to be displayed. 
    
    src_data2 : NxN numpy array
        Array containinge the data to be displayed.  
    
    scheme1 : string
        Select the color scheme with which to display src_data1.
    
    scheme2 : string
        Select the color scheme with which to display src_data2.
    
    save_file : string
        Identify whether to save the resulting figure or not
        Options:
            "YES"
            "NO"
    
    file_name : string
        String containing the file name under which to save the resulting 
        figure.
    
    Returns
    -------
    output : Two 2D plots of selected cross-sections
        Generates and displays the 2-D cross-sections, one next to the other
        Optional:
            Save the resulting figure using the specified file name
    """
    plt.figure()
    plt.subplot(211)
    plt.imshow(src_data1, cmap=scheme1)
    plt.title('Source #1 cross-section', weight= 'heavy')
    plt.subplot(212)
    plt.imshow(src_data2, cmap=scheme2)
    plt.title('Source #2 cross-section', weight='heavy')
    if  save_file == "YES":
        plt.savefig(file_name, dpi=600)
    plt.show()


def display_vol (src_data, 
                 orient = "XY"):
    """
    This function produces a scrollable viewer for the entire target volume 
    data set. The function displays 2D cross sections corresponding to the axis 
    orientation specified using the orient keyword.
    
    Parameters
    ----------
    src_data : NxNxN numpy array
        This is the source data that you want to display
        
    orient : string
        This keyword identifies the axis orientation that you want to display
        Options:
            "XY" -- display XY cross sections
            "XZ" -- display XZ cross sections
            "YZ" -- display YZ cross sections
            
    Returns
    -------
    output : popup window of scrollabe 2D cross sections.
    """
    orient_dict = {"XZ" : xform.swap_axes(src_data, "YZ"),
                   "YZ" : xform.swap_axes(src_data, "XZ"),
                   "XY" : src_data}
    disp_obj = orient_dict[orient]
    #TODO: write widgit to enable on-the-fly contrast adjustment 
    #(either max/min value setting, or slide bar)
    view_obj = viewer.CollectionViewer(disp_obj, update_on = 'move')
    view_obj.show()

def slice_select(src_data, 
                 slice_number, 
                 orientation):
    """
    Parameters
    ----------
    src_data : numpy array
        Array containing data to be sliced and depicted.
    
    slice_number : int
    
    orientation : string
        Options:
            "XY"
            "XZ"
            "YZ"
    
    Returns
    -------
    slice_selection : 2D numpy array
        Numpy array containing data to be depicted
    """
    orient_dict = {"XY" : (src_data[slice_number,:,:]),
                   "XZ" : (src_data[:,slice_number,:]),
                   "YZ" : (src_data[:,:,slice_number])}
    if len(src_data.shape) == 3:
        slice_selection = orient_dict[orientation]
    elif len(src_data.shape) == 2:
        slice_selection = src_data[:,:]
    return slice_selection
