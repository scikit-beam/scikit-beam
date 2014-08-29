# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Redistribution and use in source and binary forms, with or without   #
# modification, are permitted provided that the following conditions   #
# are met:                                                             #
#                                                                      #
# * Redistributions of source code must retain the above copyright     #
#   notice, this list of conditions and the following disclaimer.      #
#                                                                      #
# * Redistributions in binary form must reproduce the above copyright  #
#   notice this list of conditions and the following disclaimer in     #
#   the documentation and/or other materials provided with the         #
#   distribution.                                                      #
#                                                                      #
# * Neither the name of the Brookhaven Science Associates, Brookhaven  #
#   National Laboratory nor the names of its contributors may be used  #
#   to endorse or promote products derived from this software without  #
#   specific prior written permission.                                 #
#                                                                      #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS  #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT    #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS    #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE       #
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,           #
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR   #
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)   #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,  #
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OTHERWISE) ARISING   #
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE   #
# POSSIBILITY OF SUCH DAMAGE.                                          #
########################################################################

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import nsls2.recip as recip
import matplotlib as mpl
import matplotlib.pyplot as plt

spec_folder_path = "/Volumes/Data/BeamLines/Demos/X1Data/"
numpy_path = spec_folder_path + "LSCO_Oct13_numpy/"
broker_path = spec_folder_path + "LSCO_Oct13_broker/"

motors = np.load(broker_path+"motors.npy")
ub = np.load(broker_path+"ub.npy")
wavelength = np.load(broker_path+"wavelength.npy")
temp = np.load(broker_path+"temp.npy")
i_stack = np.load(broker_path+"i_stack.npy")

scan_nos = [397,450]

def reading_data():
    pass


def recip_ex():
    detector_size = (256, 256)
    pixel_size = (0.0135*8, 0.0135*8) # (mm)
    calibrated_center = (256/2.0, 256/2.0) # (mm)
    dist_sample = 355.0 # (mm)
    
    
    """wavelength = 13.27171266  # (Angstrom )

    # UB matrix (orientation matrix)
    ub_mat = np.array([[-1.397723053000000104e-01, -1.655595646000000087e+00, -1.405017159000000043e-02],
                       [-1.656324383999999927e+00, 1.398531698999999906e-01, -1.846509650999999868e-04],
                       [4.799233901999999836e-03, 4.913187235000000125e-02, -4.729227238000000000e-01]])

    # six angles of the diffractometer in following order
    # delta, theta, chi, phi, mu, gamma (in degrees)
    setting_angles = np.loadtxt("setAngles.dat", unpack = True).T"""
    for i in range(len(scan_nos)):
        ub_mat = ub[i]
        setting_angles = motors[i]
        wave_length = wavelength[i]
        I_stack = i_stack[i]
        temperature = temp[i]
        
        tot_set = recip.process_to_q(setting_angles, detector_size,
                                     pixel_size, calibrated_center,
                                     dist_sample, wave_length, ub_mat)
                                     
        print (tot_set)
        print ("#######")

        # make and ravel the image data (which is all ones)
        #I = 100 * np.ones((settingAngles.shape[0], 256, 256)).ravel()

        # minimum and maximum values of the voxel
        q_max = np.array([-0.200, 0.002, 1.410])
        q_min = np.array([-0.270, -0.002, 1.370])
        # no. of bins
        dqn = np.array([40,40,40])

        (grid_data, grid_occu, grid_std, grid_out) = recip.process_grid(tot_set, I_stack.ravel(),
                                                                        q_min, q_max, dqn=dqn)
        print (grid_data)
        run_plot(grid_occu)
    return


def run_plot(grid_occu):
    plt.figure()
    plt.imshow(grid_occu.sum(0))
    plt.show()

if __name__ == "__main__":
    recip_ex()
