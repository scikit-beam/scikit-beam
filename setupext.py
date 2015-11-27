#!/usr/bin/env python
# Copyright (c) Brookhaven National Lab 2O14
# All rights reserved
# BSD License
# See LICENSE for full text

"""
     setupext.py is for c routines

"""

from distutils.core import Extension

ctrans = {}
ctrans['define_macros'] = []
ctrans['define_macros'].append(('USE_THREADS', None))
ctrans['define_macros'].append(('NTHREADS', 1))


ext_modules = []
ext_modules.append(Extension('ctrans', ['src/ctrans.c'],
                             **ctrans))
