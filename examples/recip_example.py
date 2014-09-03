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
import nsls2.core as core
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def recip_ex():
    detector_size = (256, 256)
    pixel_size = (0.0135*8, 0.0135*8)  # (mm)
    calibrated_center = (256/2.0, 256/2.0)  # (mm)
    dist_sample = 355.0  # (mm)

    # six angles data
    motors = np.load(broker_path+"motors.npy")
    # ub matrix data
    ub = np.load(broker_path+"ub.npy")
    # wavelength data
    wavelength = np.load(broker_path+"wavelength.npy")
    # temperature data
    temp = np.load(broker_path+"temp.npy")
    # intensity of the image stack data
    i_stack = np.load(broker_path+"i_stack.npy")

    for i in range(len(scan_nos)):
        ub_mat = ub[i]
        setting_angles = motors[i]
        wave_length = wavelength[i]
        I_stack = i_stack[i]
        temperature = temp[i]

        tot_set = recip.process_to_q(setting_angles, detector_size,
                                     pixel_size, calibrated_center,
                                     dist_sample, wave_length, ub_mat)

        # minimum and maximum values of the voxel
        q_min = np.array([H_range[0], K_range[0], L_range[0]])
        q_max = np.array([H_range[1], K_range[1], L_range[1]])

        # no. of bins
        dqn = np.array([40, 40, 1])

        (grid_data, grid_occu, grid_std,
         grid_out) = recip.process_grid(tot_set, I_stack.ravel(), dqn=dqn)

        grid = np.mgrid[0:dqn[0], 0:dqn[1], 0:dqn[2]]
        r = (q_max - q_min) / dqn

        X = grid[0] * r[0] + q_min[0]
        Y = grid[1] * r[1] + q_min[1]
        Z = grid[2] * r[2] + q_min[2]

        # creating a mask
        _mask = grid_occu <= 10
        grid_Std = ma.masked_array(grid_std, _mask)
        grid_Data = ma.masked_array(grid_data, _mask)
        grid_Occu = ma.masked_array(grid_occu, _mask)

    return X, Y, Z, grid_Data, grid_Std, grid_Occu


def plottdep(ip, plane='HK'):
    grid = ip[0]

    x, y, i_slice, lx, x_ran, y_ran, x_tit, y_tit = get_xyz(grid, plane)

    i_slice_range = [0.006, 0.0115]

    f = plt.figure(1, figsize=(7.5, 2.3))
    plt.subplots_adjust(left=0.10, bottom=0.155555,
                        right=1.05, top=0.95,
                        wspace=0.2, hspace=0.45)
    subp = f.add_subplot(111)

    cnt = subp.contourf(x, y, i_slice, np.linspace(i_slice_range[0],
                                                   i_slice_range[1],
                                                   50, endpoint=True),
                        extend='both')
    subp.axis('scaled')
    subp.set_xlim(x_ran)
    subp.set_ylim(y_ran)

    subp.set_xlabel(x_tit, size=10)
    subp.set_ylabel(y_tit, size=10)

    subp.tick_params(labelsize=9)

    subp.xaxis.set_major_locator(MaxNLocator(4))
    subp.yaxis.set_major_locator(MaxNLocator(3))

    cbar = plt.colorbar(cnt, ticks=np.linspace(i_slice_range[0],
                                                 i_slice_range[1],
                                                 3, endpoint=True),
                        format='%.4f')
    cbar.ax.tick_params(labelsize=8)


def get_xyz(grid, plane):
    HKL = 'HKL'
    for i in plane:
        HKL = HKL.replace(i, '')

    HH = grid[0][:, :, :].squeeze()
    H = grid[0][:, 0, 0]
    KK = grid[1][:, :, :].squeeze()
    K = grid[1][0, :, 0]
    LL = grid[2][:, :, :].squeeze()
    L = grid[2][0, 0, :]

    i_slice = grid[3][:, :, :].squeeze()  # intensity slice
    x_ran = eval(plane[0] + '_range')  # x range
    y_ran = eval(plane[1] + '_range')  # y range

    x_tit = plane[0:1]
    y_tit = plane[1:2]

    x = eval(plane[0] + plane[0])
    y = eval(plane[1] + plane[1])
    lx = eval(plane[0])

    return x, y, i_slice, lx, x_ran, y_ran, x_tit, y_tit


if __name__ == "__main__":
    #  Data folder path
    broker_path = "LSCO_Nov12_broker/"

    # scan numbers
    scan_nos = OrderedDict()
    scan_nos["T = 015K"] = [[56]]
    scanKeys = scan_nos.keys()[0:]
    for idx, tit in enumerate(scanKeys):
        scan_no = scan_nos[tit]

    H_range = [-0.270, -0.200]
    K_range = [+0.010, -0.010]
    L_range = [+1.370, +1.410]

    ip = []
    ip.append(recip_ex())

    plottdep(ip, plane='HK')
    plt.show()
