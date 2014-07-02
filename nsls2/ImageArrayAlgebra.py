import numpy as np
import array
import Image

def ImageDataSum(Image1, Image2):
    """ 
    This module will sum two images N x N array
    Parameters: 
    ------------
    Image1 : N x N array)
             Summed image so far summed (or new image)
    Image2 : N x N array
             New image
    
    Returns:
    ---------
    SumImage : N x N array
               The image data array so far summed
    
    """
    ImgMatrix1 = np.array(Image1)
    ImgMatrix2 = np.array(Image2)
    ImageSum = ImgMatrix1 + ImgMatrix2
    return ImageSum


def DarkSubtraction(



