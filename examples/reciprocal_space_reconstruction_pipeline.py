'''
Created on May 29, 2014

@author: edill
'''


def run():
    from nsls2.io.binary import read_binary
    from nsls2.core import detector2D_to_1D
    from nsls2.recip import project_to_sphere
    import numpy as np
    from matplotlib import pyplot
    import matplotlib
    from mpl_toolkits.mplot3d import Axes3D
    fname = "nsls2/ex/data/recip_space_recon/cbr4_singlextal_rotate190_50deg_2s_90kev_203f.cor.042.cor"
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

    # list_1D = detector2D_to_1D(data, **params)

    qi = project_to_sphere(data, ROI=[900,1100,900,1100], **params)
    
    # normalize the intensity to between 0 and 1
    qi[:,3] /= max(qi[:,3])
    
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(90, 0)
    ax.scatter3D(qi[:,0], qi[:,1], qi[:,2], c=qi[:,3], s=.1)
    pyplot.show()

if __name__ == "__main__":
    run()
