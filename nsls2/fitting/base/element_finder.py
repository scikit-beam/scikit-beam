'''
Copyright (c) 2014, Brookhaven National Laboratory
All rights reserved.

# @author: Li Li (lili@bnl.gov)
# created on 08/20/2014

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

from __future__ import (absolute_import, division)
import six
import numpy as np

from nsls2.fitting.base.element import Element


def emission_line_search(incident_energy,
                         line_e, delta_e, element_list=None):
    """
    Parameters
    ----------
    incident_energy : float
        incident x-ray energy in KeV
    line_e : float
+        energy value to search for
+    delta_e : float
+        difference compared to energy
+    element_list : list
+        List of elements to search for. Element abbreviations can be
+        any mix of upper and lower case, e.g., Hg, hG, hg, HG

    Returns
    -------
    dict
        element and associate emission lines

    """
    if element_list is None:
        search_list = [Element(j + 1, incident_energy) for j in range(100)]
    else:
        search_list = [Element(item, incident_energy) for item in element_list]

    cand_lines = [e.line_near(line_e, delta_e) for e in search_list]

    out_dict = dict()
    for e, lines in zip(search_list, cand_lines):
        if lines:
            out_dict[e.name] = lines

    return out_dict
