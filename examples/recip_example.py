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
from collections import OrderedDict
import numpy.ma as ma
from pylab import *


def recip_ex():
    detector_size = (256, 256)
    pixel_size = (0.0135*8, 0.0135*8) # (mm)
    calibrated_center = (256/2.0, 256/2.0) # (mm)
    dist_sample = 355.0 # (mm)

    spec_folder_path = "/Volumes/Data/BeamLines/Demos/X1Data/"
    numpy_path = spec_folder_path + "LSCO_Oct13_numpy/"
    broker_path = spec_folder_path + "LSCO_Oct13_broker/"

    motors = np.load(broker_path+"motors.npy")
    ub = np.load(broker_path+"ub.npy")
    wavelength = np.load(broker_path+"wavelength.npy")
    temp = np.load(broker_path+"temp.npy")
    i_stack = np.load(broker_path+"i_stack.npy")

    scan_nos = [[56]]
    scan_nos = OrderedDict()
    scan_nos['E=933eV T=015K H=[-0.270,-0.200] K=[+0.002,-0.002] L=[+1.370,+1.410]'] = [[56]] # scan 56
    for idx, tit in enumerate(scan_nos.keys()[0:2]):
        H_range = [float(tit[18:24]) - 0.006, float(tit[25:31]) + 0.006]
        K_range = [float(tit[36:42]) - 0.015, float(tit[43:49]) + 0.015]
        L_range = [float(tit[54:60]) - 0.020, float(tit[61:67]) + 0.020]


    for i in range(len(scan_nos)):
        ub_mat = ub[i]
        setting_angles = motors[i]
        wave_length = wavelength[i]
        I_stack = i_stack[i]
        temperature = temp[i]
        allip = []
        tot_set = recip.process_to_q(setting_angles, detector_size,
                                     pixel_size, calibrated_center,
                                     dist_sample, wave_length, ub_mat)

        # print (tot_set)
        # print ("#######")

        # X = tot_set[:,0].ravel()
        # Y = tot_set[:,1].ravel()
        # Z = tot_set[:,2].ravel()

        q_min = np.array([H_range[0], K_range[0], L_range[0]])
        q_max = np.array([H_range[1], K_range[1], L_range[1]])
        # dqn = np.array([size, size, size])



        # minimum and maximum values of the voxel
        # q_max = np.array([max(H_range), max(K_range), max(L_range)])
        # q_min = np.array([min(H_range), min(K_range), min(L_range)])
        # no. of bins
        dqn = np.array([40,40,1])

        (grid_data, grid_occu, grid_std,
         grid_out) = recip.process_grid(tot_set, I_stack.ravel(), dqn=dqn)

        X, Y, Z = get_grid_mesh(q_min, q_max, dqn)

        # getGrid(tot_set,grid_data, grid_std, grid_occu)
        # grid1 = grid_data.ravel()
        # grid2 = grid_occu.ravel()
        # grid3 = grid_std.ravel()
        _mask = grid_occu <= 8
        grid_Std = ma.masked_array(grid_std,_mask)
        grid_Data = ma.masked_array(grid_data,_mask)
        grid_Occu = ma.masked_array(grid_occu,_mask)
        # grid1 = grid_Data.ravel()
        # grid2 = grid_occu.ravel()
        # grid3 = grid_Std.ravel()
        # allip.append(X.ravel())
        # allip.append(Y.ravel())
        # allip.append(Z.ravel())
        # allip.append(grid1.ravel())
        # allip.append(grid3.ravel())
        # allip.append(grid2.ravel())
        # I_slice = grid_data
        # grid = allip[0]
        #run_plot(grid_Data, grid_Occu)
    return X, Y, Z, grid_Data, grid_Std, grid_occu


def get_grid_mesh(Qmin, Qmax, dQN):
    """

    This function returns the X, Y and Z coordinates of the grid as 3d
    arrays. (Return the grid vectors as a mesh.
    Parameters :
    -----------
    Qmin : ndarray
        minimum values of the cuboid [Qx, Qy, Qz]_min

    Qmax : ndarray
        maximum values of the cuboid [Qx, Qy, Qz]_max

    dQN  : ndarray
        No. of grid parts (bins) [Nqx, Nqy, Nqz]

    Returns :
    --------
    X : array
        X co-ordinate of the grid

    Y : array
        Y co-ordinate of the grid

    Z : array
        Z co-ordinate of the grid


    Example :
    ---------
    These values can be used for obtaining the coordinates of each voxel.
    For instance, the position of the (0,0,0) voxel is given by

        x = X[0,0,0]
        y = Y[0,0,0]
        z = Z[0,0,0]

    """
    #h, k, l = Qmin + np.array((i, j, k)) * (QMax - Qmin) / dQN)

    grid = np.mgrid[0:dQN[0], 0:dQN[1], 0:dQN[2]]
    r = (Qmax - Qmin) / dQN

    X = grid[0] * r[0] + Qmin[0]
    Y = grid[1] * r[1] + Qmin[1]
    Z = grid[2] * r[2] + Qmin[2]

    return X, Y, Z


def plottdep(ips, tit, idx = 0, plane = 'HK'):
    grid = ips[0]

    x, y, I_Slice, lx, I_Cut, xran, yran, xtit, ytit,\
    slice_tit, cut_tit = get_xyz(grid, plane)


    I_Slice_range = [0.006, 0.0115]

    f = figure(1, figsize=(7.5, 2.3))
    subplots_adjust(left=0.10, bottom=0.155555, right=1.05, top=0.95,
                    wspace=0.2, hspace=0.45)
    subp = f.add_subplot(111)

    cnt = subp.contourf( x, y, I_Slice, linspace(I_Slice_range[0], I_Slice_range[1],
                                                 50, endpoint=True), extend='both')

    subp.axis('scaled')
    subp.set_xlim(xran)
    subp.set_ylim(yran)

    subp.set_xlabel(xtit, size=10)
    subp.set_ylabel(ytit, size=10)

    subp.tick_params( labelsize=9 )

    subp.xaxis.set_major_locator(MaxNLocator(4))
    subp.yaxis.set_major_locator(MaxNLocator(3))

    cbar = colorbar(cnt, ticks=linspace(I_Slice_range[0], I_Slice_range[1],
                                        3, endpoint=True), format='%.4f')
    cbar.ax.tick_params(labelsize=8)


def get_xyz(grid, plane):
    HKL = 'HKL'
    for i in plane: HKL = HKL.replace(i, '')

    HH = grid[0][:,:,:].squeeze(); H = grid[0][:,0,0]
    KK = grid[1][:,:,:].squeeze(); K = grid[1][0,:,0]
    LL = grid[2][:,:,:].squeeze(); L = grid[2][0,0,:]

    I_Slice = grid[3][:,:,:].squeeze()
    I_Cut = I_Slice.mean( int('HKL'.find(plane[0]) < 'HKL'.find(plane[1])) )
    print ("I_Slice")
    print (I_Slice)
    print ("#######")

    xran = eval(plane[0] + '_range')
    yran = eval(plane[1] + '_range')

    xtit = plane[0:1]; ytit = plane[1:2]
    slice_tit =  ' %s=[%s,%s]' % (HKL, eval(HKL + '_range')[0], eval(HKL + '_range')[1])
    cut_tit = slice_tit + (' %s=%s' % (plane[1], eval(plane[1] + '_range') ) )

    x  = eval(plane[0]+plane[0])
    y  = eval(plane[1]+plane[1])
    lx = eval(plane[0])

    return x, y, I_Slice, lx, I_Cut, xran, yran, xtit, ytit, slice_tit, cut_tit


def run_plot(grid_data,grid_occu):
    plt.figure()
    #plt.imshow(grid_data.sum(2))
    plt.imshow(grid_occu.sum(2))
    plt.show()


if __name__ == "__main__":
    ip = []
    ip.append(recip_ex())

    scan_nos = [[56]]
    scan_nos = OrderedDict()
    scan_nos['E=933eV T=015K H=[-0.270,-0.200] K=[+0.002,-0.002] L=[+1.370,+1.410]'] = [[56]] # scan 56
    for idx, tit in enumerate(scan_nos.keys()[0:2]):
        H_range = [float(tit[18:24]) - 0.006, float(tit[25:31]) + 0.006]
        K_range = [float(tit[36:42]) - 0.015, float(tit[43:49]) + 0.015]
        L_range = [float(tit[54:60]) - 0.020, float(tit[61:67]) + 0.020]

    plottdep(ip, tit, idx = 0, plane = 'HK')
    show()

    #X, Y, Z, grid1, grid2,grid3 = recip_ex()
