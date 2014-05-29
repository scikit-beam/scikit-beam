'''
Created on May 29, 2014

@author: edill
'''


def run():
    from nsls2.io.binary import read_binary
    from nsls2.core import detector2D_to_1D
    import numpy as np
    from matplotlib import pyplot
    fname = "/home/edill/Data/twinned cbr4/cbr4_singlextal_rotate190_50deg_2s_90kev_203f.cor.101.cor"
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
            "pixel_size": (200, 200)
            }
    # read in a binary file
    data, header = read_binary(**params)

    list_1D = detector2D_to_1D(data, **params)

    pyplot.imshow(data)

if __name__ == "__main__":
    run()
