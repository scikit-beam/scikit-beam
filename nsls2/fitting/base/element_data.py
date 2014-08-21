'''
Copyright (c) 2014, Brookhaven National Laboratory
All rights reserved.

# @author: Li Li (lili@bnl.gov)
# created on 08/16/2014

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the Brookhaven National Laboratory nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np
import six
import xraylib

line_name = ['Ka1', 'Ka2', 'Kb1', 'Kb2', 'La1', 'La2', 'Lb1', 'Lb2',
             'Lb3', 'Lb4', 'Lb5', 'Lg1', 'Lg2', 'Lg3', 'Lg4', 'Ll',
             'Ln', 'Ma1', 'Ma2', 'Mb', 'Mg']
line_list = [xraylib.KA1_LINE, xraylib.KA2_LINE, xraylib.KB1_LINE,
             xraylib.KB2_LINE, xraylib.LA1_LINE, xraylib.LA2_LINE,
             xraylib.LB1_LINE, xraylib.LB2_LINE, xraylib.LB3_LINE,
             xraylib.LB4_LINE, xraylib.LB5_LINE, xraylib.LG1_LINE,
             xraylib.LG2_LINE, xraylib.LG3_LINE, xraylib.LG4_LINE,
             xraylib.LL_LINE, xraylib.LE_LINE, xraylib.MA1_LINE,
             xraylib.MA2_LINE, xraylib.MB_LINE, xraylib.MG_LINE]

line_dict = dict((k.lower(), v) for k, v in zip(line_name, line_list))


bindingE = ['K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5', 'N1',
            'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'O1', 'O2', 'O3',
            'O4', 'O5', 'P1', 'P2', 'P3']

shell_list = [xraylib.K_SHELL, xraylib.L1_SHELL, xraylib.L2_SHELL,
              xraylib.L3_SHELL, xraylib.M1_SHELL, xraylib.M2_SHELL,
              xraylib.M3_SHELL, xraylib.M4_SHELL, xraylib.M5_SHELL,
              xraylib.N1_SHELL, xraylib.N2_SHELL, xraylib.N3_SHELL,
              xraylib.N4_SHELL, xraylib.N5_SHELL, xraylib.N6_SHELL,
              xraylib.N7_SHELL, xraylib.O1_SHELL, xraylib.O2_SHELL,
              xraylib.O3_SHELL, xraylib.O4_SHELL, xraylib.O5_SHELL,
              xraylib.P1_SHELL, xraylib.P2_SHELL, xraylib.P3_SHELL]

shell_dict = dict((k.lower(), v) for k, v in zip(bindingE, shell_list))


XRAYLIB_MAP = {'lines': (line_dict, xraylib.LineEnergy),
               'cs': (line_dict, xraylib.CS_FluorLine),
               'binding_e': (shell_dict, xraylib.EdgeEnergy),
               'jump': (shell_dict, xraylib.JumpFactor),
               'yield': (shell_dict, xraylib.FluorYield),
               }

elm_data_list = [{'Z': 1, 'mass': 1.01, 'rho': 9e-05, 'sym': 'H'},
                 {'Z': 2, 'mass': 4.0, 'rho': 0.00017, 'sym': 'He'},
                 {'Z': 3, 'mass': 6.94, 'rho': 0.534, 'sym': 'Li'},
                 {'Z': 4, 'mass': 9.01, 'rho': 1.85, 'sym': 'Be'},
                 {'Z': 5, 'mass': 10.81, 'rho': 2.34, 'sym': 'B'},
                 {'Z': 6, 'mass': 12.01, 'rho': 2.267, 'sym': 'C'},
                 {'Z': 7, 'mass': 14.01, 'rho': 0.00117, 'sym': 'N'},
                 {'Z': 8, 'mass': 16.0, 'rho': 0.00133, 'sym': 'O'},
                 {'Z': 9, 'mass': 19.0, 'rho': 0.0017, 'sym': 'F'},
                 {'Z': 10, 'mass': 20.18, 'rho': 0.00084, 'sym': 'Ne'},
                 {'Z': 11, 'mass': 22.99, 'rho': 0.97, 'sym': 'Na'},
                 {'Z': 12, 'mass': 24.31, 'rho': 1.741, 'sym': 'Mg'},
                 {'Z': 13, 'mass': 26.98, 'rho': 2.7, 'sym': 'Al'},
                 {'Z': 14, 'mass': 28.09, 'rho': 2.34, 'sym': 'Si'},
                 {'Z': 15, 'mass': 30.97, 'rho': 2.69, 'sym': 'P'},
                 {'Z': 16, 'mass': 32.06, 'rho': 2.08, 'sym': 'S'},
                 {'Z': 17, 'mass': 35.45, 'rho': 1.56, 'sym': 'Cl'},
                 {'Z': 18, 'mass': 39.95, 'rho': 0.00166, 'sym': 'Ar'},
                 {'Z': 19, 'mass': 39.1, 'rho': 0.86, 'sym': 'K'},
                 {'Z': 20, 'mass': 40.08, 'rho': 1.54, 'sym': 'Ca'},
                 {'Z': 21, 'mass': 44.96, 'rho': 3.0, 'sym': 'Sc'},
                 {'Z': 22, 'mass': 47.9, 'rho': 4.54, 'sym': 'Ti'},
                 {'Z': 23, 'mass': 50.94, 'rho': 6.1, 'sym': 'V'},
                 {'Z': 24, 'mass': 52.0, 'rho': 7.2, 'sym': 'Cr'},
                 {'Z': 25, 'mass': 54.94, 'rho': 7.44, 'sym': 'Mn'},
                 {'Z': 26, 'mass': 55.85, 'rho': 7.87, 'sym': 'Fe'},
                 {'Z': 27, 'mass': 58.93, 'rho': 8.9, 'sym': 'Co'},
                 {'Z': 28, 'mass': 58.71, 'rho': 8.908, 'sym': 'Ni'},
                 {'Z': 29, 'mass': 63.55, 'rho': 8.96, 'sym': 'Cu'},
                 {'Z': 30, 'mass': 65.37, 'rho': 7.14, 'sym': 'Zn'},
                 {'Z': 31, 'mass': 69.72, 'rho': 5.91, 'sym': 'Ga'},
                 {'Z': 32, 'mass': 72.59, 'rho': 5.323, 'sym': 'Ge'},
                 {'Z': 33, 'mass': 74.92, 'rho': 5.727, 'sym': 'As'},
                 {'Z': 34, 'mass': 78.96, 'rho': 4.81, 'sym': 'Se'},
                 {'Z': 35, 'mass': 79.9, 'rho': 3.1, 'sym': 'Br'},
                 {'Z': 36, 'mass': 83.8, 'rho': 0.00349, 'sym': 'Kr'},
                 {'Z': 37, 'mass': 85.47, 'rho': 1.53, 'sym': 'Rb'},
                 {'Z': 38, 'mass': 87.62, 'rho': 2.6, 'sym': 'Sr'},
                 {'Z': 39, 'mass': 88.91, 'rho': 4.6, 'sym': 'Y'},
                 {'Z': 40, 'mass': 91.22, 'rho': 6.5, 'sym': 'Zr'},
                 {'Z': 41, 'mass': 92.91, 'rho': 8.57, 'sym': 'Nb'},
                 {'Z': 42, 'mass': 95.94, 'rho': 10.2, 'sym': 'Mo'},
                 {'Z': 43, 'mass': 98.91, 'rho': 11.4, 'sym': 'Tc'},
                 {'Z': 44, 'mass': 101.07, 'rho': 12.4, 'sym': 'Ru'},
                 {'Z': 45, 'mass': 102.91, 'rho': 12.44, 'sym': 'Rh'},
                 {'Z': 46, 'mass': 106.4, 'rho': 12.0, 'sym': 'Pd'},
                 {'Z': 47, 'mass': 107.87, 'rho': 10.5, 'sym': 'Ag'},
                 {'Z': 48, 'mass': 112.4, 'rho': 8.65, 'sym': 'Cd'},
                 {'Z': 49, 'mass': 114.82, 'rho': 7.31, 'sym': 'In'},
                 {'Z': 50, 'mass': 118.69, 'rho': 7.3, 'sym': 'Sn'},
                 {'Z': 51, 'mass': 121.75, 'rho': 6.7, 'sym': 'Sb'},
                 {'Z': 52, 'mass': 127.6, 'rho': 6.24, 'sym': 'Te'},
                 {'Z': 53, 'mass': 126.9, 'rho': 4.94, 'sym': 'I'},
                 {'Z': 54, 'mass': 131.3, 'rho': 0.0055, 'sym': 'Xe'},
                 {'Z': 55, 'mass': 132.9, 'rho': 1.87, 'sym': 'Cs'},
                 {'Z': 56, 'mass': 137.34, 'rho': 3.6, 'sym': 'Ba'},
                 {'Z': 57, 'mass': 138.91, 'rho': 6.15, 'sym': 'La'},
                 {'Z': 58, 'mass': 140.12, 'rho': 6.8, 'sym': 'Ce'},
                 {'Z': 59, 'mass': 140.91, 'rho': 6.8, 'sym': 'Pr'},
                 {'Z': 60, 'mass': 144.24, 'rho': 6.96, 'sym': 'Nd'},
                 {'Z': 61, 'mass': 145.0, 'rho': 7.264, 'sym': 'Pm'},
                 {'Z': 62, 'mass': 150.35, 'rho': 7.5, 'sym': 'Sm'},
                 {'Z': 63, 'mass': 151.96, 'rho': 5.2, 'sym': 'Eu'},
                 {'Z': 64, 'mass': 157.25, 'rho': 7.9, 'sym': 'Gd'},
                 {'Z': 65, 'mass': 158.92, 'rho': 8.3, 'sym': 'Tb'},
                 {'Z': 66, 'mass': 162.5, 'rho': 8.5, 'sym': 'Dy'},
                 {'Z': 67, 'mass': 164.93, 'rho': 8.8, 'sym': 'Ho'},
                 {'Z': 68, 'mass': 167.26, 'rho': 9.0, 'sym': 'Er'},
                 {'Z': 69, 'mass': 168.93, 'rho': 9.3, 'sym': 'Tm'},
                 {'Z': 70, 'mass': 173.04, 'rho': 7.0, 'sym': 'Yb'},
                 {'Z': 71, 'mass': 174.97, 'rho': 9.8, 'sym': 'Lu'},
                 {'Z': 72, 'mass': 178.49, 'rho': 13.3, 'sym': 'Hf'},
                 {'Z': 73, 'mass': 180.95, 'rho': 16.6, 'sym': 'Ta'},
                 {'Z': 74, 'mass': 183.85, 'rho': 19.32, 'sym': 'W'},
                 {'Z': 75, 'mass': 186.2, 'rho': 20.5, 'sym': 'Re'},
                 {'Z': 76, 'mass': 190.2, 'rho': 22.48, 'sym': 'Os'},
                 {'Z': 77, 'mass': 192.2, 'rho': 22.42, 'sym': 'Ir'},
                 {'Z': 78, 'mass': 195.09, 'rho': 21.45, 'sym': 'Pt'},
                 {'Z': 79, 'mass': 196.97, 'rho': 19.3, 'sym': 'Au'},
                 {'Z': 80, 'mass': 200.59, 'rho': 13.59, 'sym': 'Hg'},
                 {'Z': 81, 'mass': 204.37, 'rho': 11.86, 'sym': 'Tl'},
                 {'Z': 82, 'mass': 207.17, 'rho': 11.34, 'sym': 'Pb'},
                 {'Z': 83, 'mass': 208.98, 'rho': 9.8, 'sym': 'Bi'},
                 {'Z': 84, 'mass': 209.0, 'rho': 9.2, 'sym': 'Po'},
                 {'Z': 85, 'mass': 210.0, 'rho': 6.4, 'sym': 'At'},
                 {'Z': 86, 'mass': 222.0, 'rho': 4.4, 'sym': 'Rn'},
                 {'Z': 87, 'mass': 223.0, 'rho': 2.9, 'sym': 'Fr'},
                 {'Z': 88, 'mass': 226.0, 'rho': 5.0, 'sym': 'Ra'},
                 {'Z': 89, 'mass': 227.0, 'rho': 10.1, 'sym': 'Ac'},
                 {'Z': 90, 'mass': 232.04, 'rho': 11.7, 'sym': 'Th'},
                 {'Z': 91, 'mass': 231.0, 'rho': 15.4, 'sym': 'Pa'},
                 {'Z': 92, 'mass': 238.03, 'rho': 19.1, 'sym': 'U'},
                 {'Z': 93, 'mass': 237.0, 'rho': 20.2, 'sym': 'Np'},
                 {'Z': 94, 'mass': 244.0, 'rho': 19.82, 'sym': 'Pu'},
                 {'Z': 95, 'mass': 243.0, 'rho': 12.0, 'sym': 'Am'},
                 {'Z': 96, 'mass': 247.0, 'rho': 13.51, 'sym': 'Cm'},
                 {'Z': 97, 'mass': 247.0, 'rho': 14.78, 'sym': 'Bk'},
                 {'Z': 98, 'mass': 251.0, 'rho': 15.1, 'sym': 'Cf'},
                 {'Z': 99, 'mass': 252.0, 'rho': 8.84, 'sym': 'Es'},
                 {'Z': 100, 'mass': 257.0, 'rho': np.nan, 'sym': 'Fm'}]

# make an empty dictionary
OTHER_VAL = dict()
# fill it with the data keyed on the symbol
OTHER_VAL.update((elm['sym'].lower(), elm) for elm in elm_data_list)
# also add entries with it keyed on atomic number
OTHER_VAL.update((elm['Z'], elm) for elm in elm_data_list)
