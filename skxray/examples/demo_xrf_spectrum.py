# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# @author: Li Li (lili@bnl.gov)                                        #
# created on 09/03/2014                                                #
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
from __future__ import absolute_import, division, print_function
import numpy as np

from skxray.core.constants import XrfElement
from skxray.core.fitting import gaussian


def get_line(ax, name, incident_energy):
    """
    Plot emission lines for a given element.

    Parameters
    ----------
    name : str or int
        element name, or atomic number
    incident_energy : float
        xray incident energy for fluorescence emission
    """
    e = XrfElement(name)
    lines = e.emission_line.all
    ratio = [val for val in e.cs(incident_energy).all if val[1] > 0]

    i_min = 1e-6

    for item in ratio:
        for data in lines:
            if item[0] == data[0]:
                ax.plot([data[1], data[1]],
                         [i_min, item[1]], 'g-', linewidth=2.0)

    ax.set_title('Emission lines for %s at %s eV' % (name, incident_energy))
    ax.set_xlabel('Energy [KeV]')
    ax.set_ylabel('Intensity')


def get_spectrum(ax, name, incident_energy, emax=15):
    """
    Plot fluorescence spectrum for a given element.

    Parameters
    ----------
    name : str or int
        element name, or atomic number
    incident_energy : float
        xray incident energy for fluorescence emission
    emax : float
        max value on spectrum

    """
    e = XrfElement(name)
    lines = e.emission_line.all
    ratio = [val for val in e.cs(incident_energy).all if val[1] > 0]

    x = np.arange(0, emax, 0.01)

    spec = np.zeros(len(x))

    i_min = 1e-6

    for item in ratio:
        for data in lines:
            if item[0] == data[0]:

                ax.plot([data[1], data[1]],
                         [i_min, item[1]], 'g-', linewidth=2.0)

    std = 0.1
    area = std * np.sqrt(2 * np.pi)
    for item in ratio:
        for data in lines:
            if item[0] == data[0]:
                spec += gaussian(x, area, data[1], std) * item[1]

    #plt.semilogy(x, spec)
    ax.set_title('Simulated spectrum for %s at %s eV' % (name, incident_energy))
    ax.set_xlabel('Energy [KeV]')
    ax.set_ylabel('Intensity')
    ax.plot(x, spec)


def run_demo():
    import matplotlib.pyplot as plt
    e = XrfElement('Cu')
    print('Cu ka1 = %s' % e.emission_line['ka1'])
    print('all Cu emission lines\n{}'.format(e.emission_line.all))
    print('fluorescence cross section of Cu at 12 eV = %s' % e.cs(12).all)
    print('showing spectrum for Cu at 12 eV')
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    ax = ax.ravel()
    get_line(ax[0], 'Cu', 12)
    get_spectrum(ax[1], 'Cu', 12)
    get_line(ax[2], 'Gd', 12)
    get_spectrum(ax[3], 'Gd', 12)
    plt.show()


if __name__ == "__main__":
    run_demo()
