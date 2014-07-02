# Copyright (c) Brookhaven National Lab 2O14
# All rights reserved
# BSD License
# See LICENSE for full text
"""
    This module is for Image Array Algebra
    For Summation of tiff images, Background subtraction,
    Mulifly image by a scalar factor,
    write sum/brckground subtraction image to a tiff file"""


def ImageDataSum(Image1, Image2):
    """
    Summation of images (N X N array)
    Parameters: (N X N array)
    Image1 :  The image data file (or image file so far summed)
    Image2 :  New image ( that has to sum)
    ------------------------------------------------
    Returns: (N X N array)
    ImageSum: Summation of the image data
    """
    ImageMatrix1 = np.array(Image1)
    ImageMatrix2 = np.array(Image2)
    ImageSum = ImageMatrix1 + ImageMatrix2
    return ImageSum


def ImageDataSum(Image, BackgroundImage):
    """
    Background Subtrcation of images
    Parameters: (N X N array)
    Image :  The image data file (or image file so far summed)
    BackgroundImage :  Background Image
    ------------------------------------------------
    Returns: (N X N array)
    ImageSum: Summation of the image data
    """
    ImageMatrix = np.array(Image)
    BackIMatrix = np.array(BackgroundImage)
    SubImage = ImageMatrix - BackIMatrix
    return SubImage

def write_SumImage_tif(filename, image):
    """
        Save the image data as a .tif file
        Parameters :
        image : (NxN array)
        image data
        filename : string
        name for the filename to be saved
        -------------------------------
        Returns: Saved tif file
        """
    im = Image.fromarray(image)
    im.save(filename)
    print " saved"
    return



