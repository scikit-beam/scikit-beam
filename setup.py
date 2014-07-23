#!/usr/bin/env python
# Copyright (c) Brookhaven National Lab 2O14
# All rights reserved
# BSD License
# See LICENSE for full text

import setuptools
from distutils.core import setup, Extension
from setupext import ext_modules

setup(
    name='NSLS2',
    version='0',
    author='Brookhaven National Lab',
    packages=["nsls2",
              "nsls2.testing"],
    )
