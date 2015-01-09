#! encoding: utf-8
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
This module is for the 'core' data types.
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import sys

from six import string_types
from collections import MutableMapping, defaultdict


class NotInstalledError(ImportError):
    '''
    Custom exception that should be subclassed to handle
    specific missing libraries

    '''
    pass

class MD_dict(MutableMapping):
    """
    A class to make dealing with the meta-data scheme for DataExchange easier

    Examples
    --------
    Getting and setting data by path is possible

    >>> tt = MD_dict()
    >>> tt['name'] = 'test'
    >>> tt['nested.a'] = 2
    >>> tt['nested.b'] = (5, 'm')
    >>> tt['nested.a'].value
    2
    >>> tt['nested.a'].units is None
    True
    >>> tt['name'].value
    'test'
    >>> tt['nested.b'].units
    'm'
    """
    def __init__(self, md_dict=None):
        # TODO properly walk the input on upgrade dicts -> MD_dict
        if md_dict is None:
            md_dict = dict()

        self._dict = md_dict
        self._split = '.'

    def __repr__(self):
        return self._dict.__repr__()

    # overload __setitem__ so dotted paths work
    def __setitem__(self, key, val):

        key_split = key.split(self._split)
        tmp = self._dict
        for k in key_split[:-1]:
            try:
                tmp = tmp[k]._dict
            except:
                tmp[k] = type(self)()
                tmp = tmp[k]._dict
            if isinstance(tmp, md_value):
                # TODO make message better
                raise KeyError("trying to use a leaf node as a branch")

        # if passed in an md_value, set it and return
        if isinstance(val, md_value):
            tmp[key_split[-1]] = val
            return
        # catch the case of a bare string
        elif isinstance(val, string_types):
            # a value with out units
            tmp[key_split[-1]] = md_value(val, 'text')
            return
        # not something easy, try to guess what to do instead
        try:
            # if the second element is a string or None, cast to named tuple
            if isinstance(val[1], string_types) or val[1] is None:
                print('here')
                tmp[key_split[-1]] = md_value(*val)
            # else, assume whole thing is the value with no units
            else:
                tmp[key_split[-1]] = md_value(val, None)
        # catch any type errors from trying to index into non-indexable things
        # or from trying to use iterables longer than 2
        except TypeError:
            tmp[key_split[-1]] = md_value(val, None)

    def __getitem__(self, key):
        key_split = key.split(self._split)
        tmp = self._dict
        for k in key_split[:-1]:
            try:
                tmp = tmp[k]._dict
            except:
                tmp[k] = type(self)()
                tmp = tmp[k]._dict

            if isinstance(tmp, md_value):
                # TODO make message better
                raise KeyError("trying to use a leaf node as a branch")

        return tmp.get(key_split[-1], None)

    def __delitem__(self, key):
        # pass one delete the entry
        # TODO make robust to non-keys
        key_split = key.split(self._split)
        tmp = self._dict
        for k in key_split[:-1]:
            # make sure we are grabbing the internal dict
            tmp = tmp[k]._dict
        del tmp[key_split[-1]]
        # TODO pass 2 remove empty branches

    def __len__(self):
        return len(list(iter(self)))

    def __iter__(self):
        return _iter_helper([], self._split, self._dict)


def _iter_helper(path_list, split, md_dict):
    """
    Recursively walk the tree and return the names of the leaves
    """
    for k, v in six.iteritems(md_dict):
        if isinstance(v, md_value):
            yield split.join(path_list + [k])
        else:
            for inner_v in _iter_helper(path_list + [k], split, v._dict):
                yield inner_v


class verbosedict(dict):
    """
    A sub-class of dict which raises more verbose errors if
    a key is not found.
    """
    def __getitem__(self, key):
        try:
            v = dict.__getitem__(self, key)
        except KeyError:
            if len(self) < 25:
                new_msg = ("You tried to access the key '{key}' "
                           "which does not exist.  The "
                           "extant keys are: {valid_keys}").format(
                               key=key, valid_keys=list(self))
            else:
                new_msg = ("You tried to access the key '{key}' "
                           "which does not exist.  There "
                           "are {num} extant keys, which is too many to "
                           "show you").format(
                               key=key, num=len(self))
            six.reraise(KeyError, KeyError(new_msg), sys.exc_info()[2])
        return v


class RCParamDict(MutableMapping):
    """A class to make dealing with storing default values easier.

    RC params is a hold- over from the UNIX days where configuration
    files are 'rc' files.  See
    http://en.wikipedia.org/wiki/Configuration_file

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
