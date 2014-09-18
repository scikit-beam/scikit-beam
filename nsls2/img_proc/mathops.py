# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Oct. 2013
"""
# This class is designed to facilitate image arithmetic and logical operations
# on image data sets.
"""
"""
 REVISION LOG: (FORMAT: "PROGRAMMER INITIALS: DATE -- RECORD")
 -------------------------------------------------------------
 GCI: 2/7/2014 -- Modifying documentation of the package functions for 
    inclusion in the bulk module pull to GITHUB
 GCI: 2/11/2014 -- Tom pointed out several potential issues that can be 
    caused by using python's build in logic functions. So all logic
    operations have been switch to the numpy standard function set.
 GCI: 2/18/2014 -- Updating file to proper docstring convention.
    Converting this file back to a simple module from a class structure.
    Changed file name from C2_mathops.py to mathops.py. This needs to be 
    carried through to the executable files in order for pyLight to work after 
    this update.
"""

import numpy as np


def add_img(src_data, 
            op_type, 
            offset_value):
    """
    This function enables the addition of EITHER two images or volume 
    data sets, OR an image/data set and a value. This function is typically
    used for offset purposes, or basic recombination of several isolated 
    materials or phases into a single segmented volume.

    Parameters
    ----------
    src_data : numpy array
        Specifies the data set to be offset or manipulated

    op_type : string
        Keyword to specify how the operation will be executed.
        There are currently three options for this parameter:
            manual -- will prompt the user to enter an offset value or data
                set at the command prompt>
            combine -- stipulates that a second image or data set is to be used 
                as the specified set of offset values.
            offset -- stipulates that all values in the source data set
                (src_data) are to be offset by a stipulated, pre-defined value
            offset_value -- This parameter can be set equal to EITHER the data 
                set (image) to add to source, OR a fixed value.

    Returns
    -------
    output : numpy array
        Returns the resulting array to the designated variable

    Example
    -------
    result = mathops.add_img('img_1', 'combine', 'img_2')
    """
    if op_type == 'manual':
        img_offset = input('Enter offset value:')
        output = src_data + img_offset
        print ('Volume intensity values have been offset by: ' + 
                str(img_offset))
        return output
    elif op_type == 'combine':
        output = src_data + offset_value
        print 'Volumes combined'
        return output
    elif op_type == 'offset':
        output = src_data + offset_value
        print ('Volume intensity values have been offset by: ' + 
               str(offset_value))
        return output
    else:
        print 'Addition error. Input parameters should take the form: '
        print 'add_img(src_data, op_type=offset, offset_value=0)'
        print 'where src_data is the image data to be modified'
        print 'op_type=[manual, combine, or offset]'
        print 'if manual is selected you will be directed to enter the '
        print 'offset value desired.'
        print 'if combine is selected then set offset_value equal to the '
        print '2nd image variable.'
        print 'offset is the default keywork, in which case '
        print 'offset_value equals the variable defining the desired '
        print 'offset, or is set to a fixed number.'


def sub_img(src_data, 
            op_type, 
            offset_value):
    """
    This function enables the subtraction of EITHER one image or volume data 
    set from another, OR reduction of all values in an image/data set by a set 
    value. This function is typically used for offset purposes, or basic 
    isolation of objects or materials/phases in a data set.

    Parameters
    ----------
    src_data : numpy array
        Specifies the data set to be offset or manipulated

    op_type : string
        Keyword to specify how the operation will be executed.
        There are currently three options for this parameter:
            manual -- will prompt the user to enter an offset value or data
                set at the command prompt>
            combine -- stipulates that a second image or data set is to be used 
                as the specified set of offset values.
            offset -- stipulates that all values in the source data set
                (src_data) are to be offset by a stipulated, pre-defined value
            offset_value -- This parameter can be set equal to EITHER the data 
                set (image) to add to source, OR a fixed value.

    Returns
    -------
    output : numpy array
        Returns the resulting array to the designated variable

    Example
    -------
    result = mathops.sub_img('img_1', 'combine', 'img_2')
    NOTE:
    IF The volume being evaluated is an 8-bit int data set (min=0, max=255) 
    If the subtraction value is greater than any     original cell value, then 
    the operation circles around to the maximum value of 255 and starts counting
    down.
    For Example:
        Original cell value Image1[100,100] = 2
        Subtraction offset = 5
        $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        Operation Results:
            Original value = 2
            Final Value = 253
        Because:
            2 (- 0) = 2
            2 (- 1) = 1
            2 (- 2) = 0
            2 (- 3) = 255
            2 (- 4) = 254
            2 (- 5) = 253
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    if op_type == 'manual':
        img_offset = abs(input('Enter offset value:'))
        output = src_data - img_offset
        print ('Volume intensity values have been offset by: -' + 
               str(img_offset))
        return output
    elif op_type == 'combine':
        output = src_data - offset_value
        print 'Volumes subtracted'
        return output
    elif op_type == 'offset':
        output = src_data - abs(offset_value)
        print ('Volume intensity values have been reduced by: ' + 
            str(offset_value))
        return output
    else:
        print 'Subtraction error. Input parameters should take the form: '
        print 'sub_img(src_data, op_type=offset, offset_value=0)'
        print 'where src_data is the image data to be modified'
        print 'op_type=[manual, combine, or offset]'
        print 'if manual is selected you will be directed to enter the '
        print 'offset value desired.'
        print 'if combine is selected then set offset_value equal to the '
        print '2nd image variable.'
        print 'offset is the default keywork, in which case '
        print 'offset_value equals the variable defining the desired '
        print 'offset, or is set to a fixed number.'


def mult_img(src_data, 
             op_type, 
             offset_value):
    """
    This function allows the multiplication of EITHER one image or volume 
    data set with another, OR multiplication of all values in an image/data set
    by a set value. This function is typically used to increase the 
    distribution of the volume histogram. For example: a volume with peaks at 0, 
    10 and 25, multiplied by 5 would result in a volume with peaks at 0, 50, and
    125.
    This operation can be very useful in segmentation or phase seperation and 
    is expected to be used often in our image processing routines.

    Parameters
    ----------
    src_data : numpy array
        Specifies the data set to be offset or manipulated

    op_type : string
        Keyword to specify how the operation will be executed.
        There are currently three options for this parameter:
            manual -- will prompt the user to enter an offset value or data
                set at the command prompt>
            combine -- stipulates that a second image or data set is to be used 
                as the specified set of offset values.
            offset -- stipulates that all values in the source data set
                (src_data) are to be offset by a stipulated, pre-defined value
            offset_value -- This parameter can be set equal to EITHER the data 
                set (image) to add to source, OR a fixed value.

    Returns
    -------
    output : numpy array
        Returns the resulting array to the designated variable

    Example
    -------
    result = mathops.mult_img('img_1', 'combine', 'img_2')
    """
    if op_type == 'manual':
        img_offset = abs(input('Enter offset value:'))
        output = src_data * img_offset
        print ('Volume intensity values have been multiplied by: -' + 
               str(img_offset))
        return output
    elif op_type == 'combine':
        output = src_data * offset_value
        print 'Volumes subtracted'
        return output
    elif op_type == 'offset' and offset_value != 0:
        output = src_data * offset_value
        print ('Volume intensity values have been multiplied by: ' + 
               str(offset_value))
        return output
    elif op_type == 'offset' and offset_value == 0:
        print 'Multiplier equals 0! Are you sure you wish to continue?'
        answer = input('Are you sure you with to continue?: (y/n)')
        if answer == 'y' or answer == 'Y' or answer == 'Yes' or answer == 'YES' or answer == 'yes':
            output = src_data * offset_value
            return output
        else:
            print 'Operation cancelled due to zero value multiplier.'
            # not sure how to exit out of the if statement. the print statement 
            # is in as a filler.
    else:
        print 'Multiplication error. Input parameters should take the form: '
        print 'mult_img(src_data, op_type=offset, offset_value=0)'
        print 'where src_data is the image data to be modified'
        print 'op_type=[manual, combine, or offset]'
        print 'if manual is selected you will be directed to enter the '
        print 'offset value desired.'
        print 'if combine is selected then set offset_value equal to the '
        print '2nd image variable.'
        print 'offset is the default keywork, in which case '
        print 'offset_value equals the variable defining the desired '
        print 'offset, or is set to a fixed number.'


def div_img(src_data, 
            op_type, 
            offset_value):
    """
    This function enables the division of EITHER one image or volume data set by 
    another, OR division of all values in an image/data set by a set value. 
    This function is typically used for data set normalization purposes, or 
    basic  isolation of objects or materials/phases in a data set.

    Parameters
    ----------
    src_data : numpy array
        Specifies the data set to be offset or manipulated

    op_type : string
        Keyword to specify how the operation will be executed.
        There are currently three options for this parameter:
            manual -- will prompt the user to enter an offset value or data
                set at the command prompt>
            combine -- stipulates that a second image or data set is to be used 
                as the specified set of offset values.
            offset -- stipulates that all values in the source data set
                (src_data) are to be offset by a stipulated, pre-defined value
            offset_value -- This parameter can be set equal to EITHER the data 
                set (image) to add to source, OR a fixed value.

    Returns
    -------
    output : numpy array
        Returns the resulting array to the designated variable

    Example
    -------
    result = mathops.div_img('img_1', 'combine', 'img_2')
    """
    if op_type == 'manual':
        img_offset = input('Enter offset value:')
        output = src_data / img_offset
        print ('Volume intensity values have been divided by: -' + 
               str(img_offset))
        return output
    elif op_type == 'combine':
        output = src_data / offset_value
        print 'Volumes divided'
        return output
    elif op_type == 'offset' and offset_value != 0:
        output = src_data / abs(offset_value)
        print ('Volume intensity values have been divided by: ' + 
               str(offset_value))
        return output
    elif op_type == 'offset' and offset_value == 0:
        print 'Denominator equals 0! Operation will not continue.'
    else:
        print 'Subtraction error. Input parameters should take the form: '
        print 'sub_img(src_data, op_type=offset, offset_value=0)'
        print 'where src_data is the image data to be modified'
        print 'op_type=[manual, combine, or offset]'
        print 'if manual is selected you will be directed to enter the '
        print 'offset value desired.'
        print 'if combine is selected then set offset_value equal to the '
        print '2nd image variable.'
        print 'offset is the default keywork, in which case '
        print 'offset_value equals the variable defining the desired '
        print 'offset, or is set to a fixed number.'


def logical_AND(src_data1, 
                src_data2):
    """
    This function enables the computation of the LOGICAL_AND of two image or 
    volume  data sets. This function can be used for data comparison, material 
    isolation, noise removal, or mask application/generation.
    NOTE: 
    The actual operator in the fuction can be the bitwise operator "&" as in 
    "x & y = z". The function also works using the standard logical operator 
    "and" as in "x and y = z".

    Parameters
    ----------
    src_data1 : numpy array
        Specifies the first reference data

    src_data2 : numpy array
        Specifies the second reference data

    Returns
    -------
    output : BOOL numpy array
        Returns the resulting array to the designated variable

    Example
    -------
    result = mathops.logical_AND('img_1', 'img_2')
	"""
    output = np.logical_and(src_data1, 
                            src_data2)
    print 'Operation successful: ' + src_data1 + ' AND ' + src_data2
    return output


def logical_OR(src_data1, 
               src_data2):
    """
    This function enables the computation of the LOGICAL_OR of two image or 
    volume data sets. This function can be used for data comparison, 
    material isolation, noise removal, or mask application/generation.
    NOTE:
    The actual operator in the fuction can be the bitwise operator "|" as in 
    "x | y = z". The function also works using the standard logical operator 
    "or" as in "x or y = z".


    Parameters
    ----------
    src_data1 : numpy array
        Specifies the first reference data

    src_data2 : numpy array
        Specifies the second reference data

    Returns
    -------
    output : BOOL numpy array
        Returns the resulting array to the designated variable

    Example
    -------
    result = mathops.logical_OR('img_1', 'img_2')
    """
    output = np.logical_or(src_data1, 
                           src_data2)
    print 'Operation successful: ' + src_data1 + ' OR ' + src_data2
    return output


def logical_NOT(src_data):
    """
    This function enables the computation of the LOGICAL_NOT of an image or 
    volume data set. This function can be used for data comparison, material 
    isolation, noise removal, or mask application/generation, and operates as a 
    simple inversion of a binary data set.
    NOTE:
    The actual operator in the fuction can be the bitwise operator "~" as in 
    "~x = z". The function also works using the standard logical operator "not" 
    as in "not x = z"

    Parameters
    ----------
    src_data1 : binary or BOOL numpy array
        Specifies the reference data to be 'inverted'

    Returns
    -------
    output : BOOL numpy array
        Returns the resulting array to the designated variable

    Example
    -------
    result = mathops.logical_NOT('img_1')
    """
    output = np.logical_not(src_data)
    print 'Operation successful: ' + 'NOT ' + src_data
    return output


def logical_NAND(src_data1, 
                 src_data2):
    """
    This function enables the computation of the LOGICAL_NAND of two image or 
    volume data sets. This function enables easy isolation of all data points 
    NOT INCLUDED IN BOTH SOURCE DATA SETS. This function can be used for data 
    comparison, material isolation, noise removal, or mask 
    application/generation.

    Parameters
    ----------
    src_data1 : numpy array
        Specifies the first reference data

    src_data2 : numpy array
        Specifies the second reference data

    Returns
    -------
    output : BOOL numpy array
        Returns the resulting array to the designated variable

    Example
    -------
    result = mathops.logical_NAND('img_1', 'img_2')
    """
    output = np.logical_not(np.logical_and(src_data1, 
                                           src_data2))
    return output


def logical_XOR(src_data1, 
                src_data2):
    """
    This function enables the computation of the LOGICAL_XOR of two image or 
    volume data sets. This function will evaluate as true for all "true" 
    voxels located in only one or the other data set. This function can be used 
    for data comparison, material isolation, noise removal, or mask 
    application/generation.
    NOTE:
    The actual operator in the fuction defenition can be "^" as in "x ^ y = z"

    Parameters
    ----------
    src_data1 : numpy array
        Specifies the first reference data

    src_data2 : numpy array
        Specifies the second reference data

    Returns
    -------
    output : BOOL numpy array
        Returns the resulting array to the designated variable

    Example
    -------
    result = mathops.logical_XOR('img_1', 'img_2')
    """
    output = np.logical_xor(src_data1, 
                            src_data2)
    return output


def logical_SUB(src_data1, 
                src_data2):
    """
    This function enables LOGICAL SUBTRACTION of one binary image or volume data 
    set from another. This function can be used to remove phase information, 
    interface boundaries, or noise, present in two data sets, without having to 
    worry about mislabeling of pixels which would result from arithmetic 
    subtraction. This function will evaluate as true for all "true" voxels 
    present ONLY in Source Dataset 1. This function can be used for data 
    cleanup, or boundary/interface analysis.

    Parameters
    ----------
    src_data1 : numpy array
        Specifies the first reference data

    src_data2 : numpy array
        Specifies the second reference data

    Returns
    -------
    output : BOOL numpy array
        Returns the resulting array to the designated variable

    Example
    -------
    result = mathops.logical_SUB('img_1', 'img_2')
    """
    temp = np.logical_not(np.logical_and(src_data1, 
                                         src_data2))
    output = np.logical_and(src_data1, 
                            temp)
    return output


#def extract_phase(src_data, 
                  #material_value):
    #"""
    #This function allows the extraction of all voxels that have the 
    #defined intensity value. Typically this function will be utilized in order 
    #to isolate a single material from a segmented data set for additional 
    #visualization, manipulation or comparison.

    #Parameters
    #----------
    #src_data : numpy array
        #Specifies the first reference data

    #material_value : integer
        #Specifies value of all voxels that you want to extract from the data set 
        #into a seperate binary volume.

    #Returns
    #-------
    #output : binary numpy array
        #Returns the resulting array to the designated variable

    #Example
    #-------
    #result = mathops.extract_phase('img_1', 2)

    #Example details:
    #The source volume contains a total of 3 materials plus the sample exterior.
        #Solid Phase has been assigned a value of 2
        #Oil Phase has been assigned a value of 3
        #Water Phase has been assigned a value of 1
        #Exterior has been assigned a value of 0

    #Example result:
    #Selecting a Value of 2 will produce a single binary data set which contains 
    #only voxels associated with the solid phase (assigned a value of 1) and all 
    #other voxels are assigned a value of 0. Final Result is a BINARY data set
    #"""
    #output = src_data == material_value
    #print ('Material ' + str(material_value) + ' has been successfully 
        #extracted from the source data set.')
    #return output


#def extract_allElse(src_data, 
                    #material_value):
    #"""
    #This function allows the extraction of all voxels that do not have the 
    #defined intensity value. Typically this function will be utilized in order 
    #to isolate a single material from a segmented data set for additional 
    #visualization, manipulation or comparison.

    #Parameters
    #----------
    #src_data : numpy array
        #Specifies the first reference data
    #material_value : integer
        #Specifies value of all voxels that you want to exclude from the 
        #resulting binary volume.

    #Returns
    #-------
    #output : binary numpy array
        #Returns the resulting array to the designated variable

    #Example
    #-------
    #result = mathops.extract_allElse('img_1', 2)
    
    #Example details:
    #The source volume contains a total of 3 materials plus the sample exterior.
        #Solid Phase has been assigned a value of 2
        #Oil Phase has been assigned a value of 3
        #Water Phase has been assigned a value of 1
        #Exterior has been assigned a value of 0
    #Example result:
    #Selecting a Value of 2 will produce a single binary data set which contains 
    #only voxels associated with the pore space and exterior (assigned a value of 
    #1) and all other voxels are assigned a value of 0. Final Result is a BINARY 
    #data set
    #"""
    #output = src_data != material_value
    #print ('Material ' + str(material_value) + ' has been successfully 
    #extracted from the source data set.')
    #return output
