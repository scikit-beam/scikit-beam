'''
Copyright (c) 2014, Brookhaven National Laboratory
All rights reserved.

# @author: Li Li (lili@bnl.gov)
# created on 08/12/2014

Original code:
@author: Mirna Lerotic, 2nd Look Consulting
         http://www.2ndlookconsulting.com/
Copyright (c) 2013, Stefan Vogt, Argonne National Laboratory
All rights reserved.

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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import csv
import os
import logging

import xraylib



class element_info:
    """
    information related to element fluorescence
    """
    def __init__(self):
        """
        Parameters:
        ----------
        self.z : int
            atomic number
        self.name : string
            element name
        self.xrf : dict
            all the emission lines
        self.xrf_abs_yield : dict
            all the x-ray fluorescence cross section, unit cm2/g
        self.yieldD : dict
            yield for k, l1, l2, l3 and m shell
        self.density : float
            element density in r.t.
        self.mass : float
            atomic mass
        self.bindingE : dict
            binding energy for different shells
        self.jump : dict
            jump factor for different shells
        """
        self.z = 0
        self.name = ''
        self.xrf = {'Ka1':0., 'Ka2':0.,
                    'Kb1':0., 'Kb2':0.,
                    'La1':0., 'La2':0., 'Lb1':0., 'Lb2':0., 'Lb3':0., 'Lb4':0., 'Lb5':0.,
                    'Lg1':0., 'Lg2':0., 'Lg3':0., 'Lg4':0., 'Ll':0., 'Ln':0.,
                    'Ma1':0., 'Ma2':0., 'Mb':0., 'Mg':0. }
        self.xrf_abs_yield = {'Ka1':0., 'Ka2':0.,
                              'Kb1':0., 'Kb2':0.,
                              'La1':0., 'La2':0., 'Lb1':0., 'Lb2':0., 'Lb3':0., 'Lb4':0., 'Lb5':0.,
                              'Lg1':0., 'Lg2':0., 'Lg3':0., 'Lg4':0., 'Ll':0., 'Ln':0.,
                              'Ma1':0., 'Ma2':0., 'Mb':0., 'Mg':0. }
        self.yieldD = {'k':0., 'l1':0., 'l2':0., 'l3':0., 'm':0. }
        self.density = 1.
        self.mass = 1.
        self.bindingE = {'K':0.,
                         'L1':0., 'L2':0., 'L3':0.,
                         'M1':0., 'M2':0., 'M3':0., 'M4':0., 'M5':0.,
                         'N1':0., 'N2':0., 'N3':0., 'N4':0., 'N5':0., 'N6':0., 'N7':0.,
                         'O1':0., 'O2':0., 'O3':0., 'O4':0., 'O5':0.,
                         'P1':0., 'P2':0., 'P3':0. }
        self.jump = {'K':0.,
                     'L1':0., 'L2':0., 'L3':0.,
                     'M1':0., 'M2':0., 'M3':0., 'M4':0., 'M5':0.,
                     'N1':0., 'N2':0., 'N3':0., 'N4':0., 'N5':0.,
                     'O1':0., 'O2':0., 'O3':0. }




def get_element_info(nels=100,
                     filenam='xrf_library.csv'):
    """
    get element fluorescence information from file
    
    Parameters:
    ----------
    nels : int
        number of elements saved in the file
    filename : string
        file saving all the elements
    Returns:
    --------
    element : class object
        save all the elements fluorescence information
    """
    
    file_dir = os.path.dirname(__file__)
    els_file = os.path.join(file_dir, )
    
    try:
        f = open(els_file, 'r')
        csvf = csv.reader(f, delimiter=',')
    except IOError:
        errmsg = 'Error: Could not find file %s!' % (filename)
        print (errmsg)
        logging.critical(errmsg)


    element = []
    for i in range(nels):
            element.append(element_info())


    rownum = 1 #skip header
    for row in csvf:
        if (row[0]=='version:') or (row[0]=='') or \
            (row[0]=='aprrox intensity') or (row[0]=='transition') or \
            (row[0]=='Z') :
            continue
        i = int(row[0])-1

        element[i].z = int(float(row[0]))
        element[i].name = row[1]
        element[i].xrf['ka1'] = float(row[2])
        element[i].xrf['ka2'] = float(row[3])
        element[i].xrf['kb1'] = float(row[4])
        element[i].xrf['kb2'] = float(row[5])
        element[i].xrf['la1'] = float(row[6])
        element[i].xrf['la2'] = float(row[7])
        element[i].xrf['lb1'] = float(row[8])
        element[i].xrf['lb2'] = float(row[9])
        element[i].xrf['lb3'] = float(row[10])
        element[i].xrf['lb4'] = float(row[11])
        element[i].xrf['lg1'] = float(row[12])
        element[i].xrf['lg2'] = float(row[13])
        element[i].xrf['lg3'] = float(row[14])
        element[i].xrf['lg4'] = float(row[15])
        element[i].xrf['ll'] = float(row[16])
        element[i].xrf['ln'] = float(row[17])
        element[i].xrf['ma1'] = float(row[18])
        element[i].xrf['ma2'] = float(row[19])
        element[i].xrf['mb'] = float(row[20])
        element[i].xrf['mg'] = float(row[21])
        element[i].yieldD['k'] = float(row[22])
        element[i].yieldD['l1'] = float(row[23])
        element[i].yieldD['l2'] = float(row[24])
        element[i].yieldD['l3'] = float(row[25])
        element[i].yieldD['m'] = float(row[26])
        element[i].xrf_abs_yield['ka1'] = float(row[27])
        element[i].xrf_abs_yield['ka2'] = float(row[28])
        element[i].xrf_abs_yield['kb1'] = float(row[29])
        element[i].xrf_abs_yield['kb2'] = float(row[30])
        element[i].xrf_abs_yield['la1'] = float(row[31])
        element[i].xrf_abs_yield['la2'] = float(row[32])
        element[i].xrf_abs_yield['lb1'] = float(row[33])
        element[i].xrf_abs_yield['lb2'] = float(row[34])
        element[i].xrf_abs_yield['lb3'] = float(row[35])
        element[i].xrf_abs_yield['lb4'] = float(row[36])
        element[i].xrf_abs_yield['lg1'] = float(row[37])
        element[i].xrf_abs_yield['lg2'] = float(row[38])
        element[i].xrf_abs_yield['lg3'] = float(row[39])
        element[i].xrf_abs_yield['lg4'] = float(row[40])
        element[i].xrf_abs_yield['ll'] = float(row[41])
        element[i].xrf_abs_yield['ln'] = float(row[42])
        element[i].xrf_abs_yield['ma1'] = float(row[43])
        element[i].xrf_abs_yield['ma2'] = float(row[44])
        element[i].xrf_abs_yield['mb'] = float(row[45])
        element[i].xrf_abs_yield['mg'] = float(row[46])

        if len(row) > 46 :
            element[i].density = float(row[47])
            element[i].mass = float(row[48])

            element[i].bindingE['K'] = float(row[49])

            element[i].bindingE['L1'] = float(row[50])
            element[i].bindingE['L2'] = float(row[51])
            element[i].bindingE['L3'] = float(row[52])

            element[i].bindingE['M1'] = float(row[53])
            element[i].bindingE['M2'] = float(row[54])
            element[i].bindingE['M3'] = float(row[55])
            element[i].bindingE['M4'] = float(row[56])
            element[i].bindingE['M5'] = float(row[57])

            element[i].bindingE['N1'] = float(row[58])
            element[i].bindingE['N2'] = float(row[59])
            element[i].bindingE['N3'] = float(row[60])
            element[i].bindingE['N4'] = float(row[61])
            element[i].bindingE['N5'] = float(row[62])
            element[i].bindingE['N6'] = float(row[63])
            element[i].bindingE['N7'] = float(row[64])

            element[i].bindingE['O1'] = float(row[65])
            element[i].bindingE['O2'] = float(row[66])
            element[i].bindingE['O3'] = float(row[67])
            element[i].bindingE['O4'] = float(row[68])
            element[i].bindingE['O5'] = float(row[69])

            element[i].bindingE['P1'] = float(row[70])
            element[i].bindingE['P2'] = float(row[71])
            element[i].bindingE['P3'] = float(row[72])


            element[i].jump['K'] = float(row[73])

            element[i].jump['L1'] = float(row[74])
            element[i].jump['L2'] = float(row[75])
            element[i].jump['L3'] = float(row[76])

            element[i].jump['M1'] = float(row[77])
            element[i].jump['M2'] = float(row[78])
            element[i].jump['M3'] = float(row[79])
            element[i].jump['M4'] = float(row[80])
            element[i].jump['M5'] = float(row[81])

            element[i].jump['N1'] = float(row[82])
            element[i].jump['N2'] = float(row[83])
            element[i].jump['N3'] = float(row[84])
            element[i].jump['N4'] = float(row[85])
            element[i].jump['N5'] = float(row[86])

            element[i].jump['O1'] = float(row[87])
            element[i].jump['O2'] = float(row[88])
            element[i].jump['O3'] = float(row[89])      


    f.close()

    return element


def get_element_xraylib(incident_energy=10.0,
                        filename='element_data.dat'):
    """
    get all the elements information from xraylib
    the cross section is energy dependent
    
    Parameters:
    ----------
    incident_energy : float
        incident x-ray energy to emit fluorescence line
    filename : string
        file saving the element name, density, mass
    Returns:
    --------
    element : class object
        save all the elements fluorescence information
    """
    
    file_dir = os.path.dirname(__file__)
    myfile = os.path.join(file_dir, filename)
    
    try:
        f = open(myfile, 'r')
    except IOError:
        errmsg = 'Error: Could not find file %s!' % (filename)
        print (errmsg)
        logging.critical(errmsg)


    lines = f.readlines()

    
    # for xraylib format
    line_list = [xraylib.KA1_LINE, xraylib.KA2_LINE, xraylib.KB1_LINE, xraylib.KB2_LINE, 
                 xraylib.LA1_LINE, xraylib.LA2_LINE, 
                 xraylib.LB1_LINE, xraylib.LB2_LINE, xraylib.LB3_LINE, xraylib.LB4_LINE, xraylib.LB5_LINE,
                 xraylib.LG1_LINE, xraylib.LG2_LINE, xraylib.LG3_LINE, xraylib.LG4_LINE, 
                 xraylib.LL_LINE, xraylib.LE_LINE,
                 xraylib.MA1_LINE, xraylib.MA2_LINE, xraylib.MB_LINE, xraylib.MG_LINE]

    shell_list = [xraylib.K_SHELL, xraylib.L1_SHELL, xraylib.L2_SHELL, xraylib.L3_SHELL,
                  xraylib.M1_SHELL, xraylib.M2_SHELL, xraylib.M3_SHELL, xraylib.M4_SHELL, xraylib.M5_SHELL,
                  xraylib.N1_SHELL, xraylib.N2_SHELL, xraylib.N3_SHELL, xraylib.N4_SHELL, 
                  xraylib.N5_SHELL, xraylib.N6_SHELL, xraylib.N7_SHELL,
                  xraylib.O1_SHELL, xraylib.O2_SHELL, xraylib.O3_SHELL, xraylib.O4_SHELL, xraylib.O5_SHELL,
                  xraylib.P1_SHELL, xraylib.P2_SHELL, xraylib.P3_SHELL]
    
    jump_list = [xraylib.K_SHELL, xraylib.L1_SHELL, xraylib.L2_SHELL, xraylib.L3_SHELL,
                 xraylib.M1_SHELL, xraylib.M2_SHELL, xraylib.M3_SHELL, xraylib.M4_SHELL, xraylib.M5_SHELL,
                 xraylib.N1_SHELL, xraylib.N2_SHELL, xraylib.N3_SHELL, xraylib.N4_SHELL, xraylib.N5_SHELL,
                 xraylib.O1_SHELL, xraylib.O2_SHELL, xraylib.O3_SHELL]
    
    
    element = []
    for i in np.arange(len(lines)):
        element.append(element_info())


    for i in np.arange(0, len(lines)):
        
        lines[i] = lines[i].strip()
        myline = lines[i].split()
        
        # ignore the first line
        if myline[0] == 'name': continue
        
        
        element[i].z = i
        element[i].name = myline[0]
        
        element[i].density = float(myline[1])
        element[i].mass = float(myline[2])
        
        # emission line energy and Fluorescence cross section
        keys = element[i].xrf.keys()
        keys.sort()
        for j in np.arange(len(keys)):
                element[i].xrf[keys[j]] = xraylib.LineEnergy(i, line_list[j])
                element[i].xrf_abs_yield[keys[j]] = xraylib.CS_FluorLine(i, line_list[j], incident_energy)
        
        # binding energy 
        keys = element[i].bindingE.keys()
        keys.sort()
        for j in np.arange(len(keys)):
            element[i].bindingE[keys[j]] = xraylib.EdgeEnergy(i, shell_list[j])
    
        # jump factor
        keys = element[i].jump.keys()
        keys.sort()
        for j in np.arange(len(keys)):
            element[i].jump[keys[j]] = xraylib.JumpFactor(i, shell_list[j])
    
        # yield for several main lines
        keys = element[i].yieldD.keys()
        keys.sort()
        for j in np.arange(len(keys)):
            element[i].yieldD[keys[j]] = xraylib.FluorYield(i, shell_list[j])
    
    f.close()

    return element

    

