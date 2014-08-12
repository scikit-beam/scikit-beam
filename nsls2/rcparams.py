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
"""
This module is for dealing with keeping track of package-wide defaults
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

from collections import MutableMapping, defaultdict
import logging
logger = logging.getLogger(__name__)

class RCParamDict(MutableMapping):
    """
    A class to make dealing with storing RC params easier.  RC params is a hold-
    over from the UNIX days where configuration files are 'rc' files.
    See http://en.wikipedia.org/wiki/Configuration_file

    Examples
    --------
    Getting and setting data by path is possible

    >>> tt = RCParamDict()
    >>> tt['name'] = 'test'
    >>> tt['nested.a'] = 2
    """
    _delim = '.'

    def __init__(self):
        # the dict to hold the keys at this level
        self._dict = dict()
        # the defaultdict (defaults to just accepting it) of
        # validator functions
        self._validators = defaultdict(lambda: lambda x: True)

    # overload __setitem__ so dotted paths work
    def __setitem__(self, key, val):
        # try to split the key
        splt_key = key.split(self._delim, 1)
        # if more than one part, recurse
        if len(splt_key) > 1:
            try:
                tmp = self._dict[splt_key[0]]
            except KeyError:
                tmp = RCParamDict()
                self._dict[splt_key[0]] = tmp

            if not isinstance(tmp, RCParamDict):
                raise KeyError("name space is borked")

            tmp[splt_key[1]] = val
        else:
            if not self._validators[key]:
                # TODO improve the validation error
                raise ValueError("fails to validate, improve this")
            self._dict[key] = val

    def __getitem__(self, key):
        # try to split the key
        splt_key = key.split(self._delim, 1)
        if len(splt_key) > 1:
            return self._dict[splt_key[0]][splt_key[1]]
        else:
            return self._dict[key]

    def __delitem__(self, key):
        splt_key = key.split(self._delim, 1)
        if len(splt_key) > 1:
            self._dict[splt_key[0]].__delitem__(splt_key[1])
        else:
            del self._dict[key]

    def __len__(self):
        return len(list(iter(self)))

    def __iter__(self):
        return self._iter_helper([])

    def _iter_helper(self, path_list):
        """
        Recursively walk the tree and return the names of the leaves
        """
        for key, val in six.iteritems(self._dict):
            if isinstance(val, RCParamDict):
                for k in val._iter_helper(path_list + [key, ]):
                    yield k
            else:
                yield self._delim.join(path_list + [key, ])

    def __repr__(self):
        # recursively get the formatted list of strings
        str_list = self._repr_helper(0)
        # return as a single string
        return '\n'.join(str_list)

    def _repr_helper(self, tab_level):
        # to accumulate the strings into
        str_list = []
        # list of the elements at this level
        elm_list = []
        # list of sub-levels
        nested_list = []
        # loop over the local _dict and sort out which
        # keys are nested and which are this level
        for key, val in six.iteritems(self._dict):
            if isinstance(val, RCParamDict):
                nested_list.append(key)
            else:
                elm_list.append(key)

        # sort the keys in both lists
        elm_list.sort()
        nested_list.sort()

        # loop over and format the keys/vals at this level
        for elm in elm_list:
            str_list.append("    " * tab_level +
                            "{key}: {val}".format(
                                key=elm, val=self._dict[elm]))
        # deal with the nested groups
        for nested in nested_list:
            # add the label for the group name
            str_list.append("    " * tab_level +
                            "{key}:".format(key=nested))
            # add the strings from _all_ the nested groups
            str_list.extend(
                self._dict[nested]._repr_helper(tab_level + 1))
        return str_list
