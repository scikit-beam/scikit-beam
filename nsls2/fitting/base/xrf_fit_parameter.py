'''
Copyright (c) 2014, Brookhaven National Laboratory
All rights reserved.

# @author: Li Li (lili@bnl.gov)
# created on 08/07/2014

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

from __future__ import (absolute_import, division, 
                        print_function, unicode_literals)

import os
import numpy as np
import logging

from collections import OrderedDict
#logging.basicConfig(filename='file.log', level=logging.DEBUG)


class FittingParameter(object):
    """
    basic data structure to save general fitting parameters
    """
    def __init__(self, value, 
                 use, min, max, 
                 option_list):
        """
        Parameters:
        -----------
        value : float
            value of the parameter
        use : int
            option to define if the fitting parameter has bounds or not
        min : float
            lower bound
        max : float
            higher bound
        option_list : list
            other choices to control the bounds of fitting parameters
        """
        self.value = value
        self.use = use
        self.min = min
        self.max = max
        self.option_list = option_list



def get_parameters(filename = 'parameter_general.txt'):
    """
    return the dictionary which saves all fitting parameters
    
    Parameters:
    ----------
    filename : string
        filename saving defaulted parameters
    
    Returns:
    --------
    para : dict
        dict saving all fitting parameters
    """
    file_dir = os.path.dirname(__file__)
    filepath = os.path.join(file_dir, filename)

    try:
        myfile = open(filepath, 'r')
    except IOError:
        err_msg = "No default parameter file: %s " % filename
        print (err_msg)
        logging.error(str(err_msg))
    
    para = {}
    
    lines = myfile.readlines()
    
    title = lines[0].split('\t')
    title = [str(item.strip('\n')) for item in title]
    
    logging.info('Started saving general parameters as dictionary')
    #for item in title:
    #    para[item] = 10
    for i in np.arange(1, len(lines)):
        line = lines[i].split('\t')
        name = str(line[0])
        line_val = line[1:]
        line_val = [float(item.strip('\n')) for item in line_val]
        obj = FittingParameter(line_val[0], line_val[1], line_val[2],
                               line_val[3], line_val[4:])

        para[name] = obj

    logging.info('Finished saving general parameters as dictionary')
    
    return para


def transfer_dict(p):
    pnew = {}
    for k, v in p.items():
        print (k,v)

        dict0 = dict(value=v.value, use=v.use,
                     min=v.min, max=v.max,
                     option0=v.option_list[0],
                     option1=v.option_list[1],
                     option2=v.option_list[2],
                     option3=v.option_list[3],
                     option4=v.option_list[4])

        newdict = {k: dict0}
        pnew.update(newdict)
    return pnew


p = get_parameters()
print (p['si_escape'].option_list)
    
pnew = transfer_dict(p)

print (pnew)



