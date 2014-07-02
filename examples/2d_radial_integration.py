'''
Created on Jun 4, 2014

@author: Eric-t61p
'''

import matplotlib as mpl
import numpy as np
from nsls2.io.binary import read_binary
from nsls2.core import detector2D_to_1D

def get_cbr4_sample_img():
    # define the 
    fname = "data/2d/cbr4_singlextal_rotate190_50deg_2s_90kev_203f.cor.042.cor"
    params = {"filename": fname,
            "nx": 2048,
            "ny": 2048,
            "nz": 1,
            "headersize": 0,
            "dsize": np.uint16,
            # these numbers come from https://github.com/JamesDMartin/RamDog/blob/master/Calibration/APS--2009--CeO2.calib
            "wavelength": .13702,
            "detector_center": (1033.321, 1020.208),
            "dist_sample": 188.672,
            "pixel_size": (.200, .200)
            }
    
    # read in a binary file
    data, header = read_binary(**params)
    
    return data, params
    
def run():
    # get the sample data
    data, params = get_cbr4_sample_img
    # convert the data from 2d array to xyi relative to beam center
    xyi = detector2D_to_1D(data, **params)
    # convert xy to r
    r = np.linalg.norm(xyi[:,0:2])
    # bin i based on r
    
    
if __name__ == "__main__":
    run()