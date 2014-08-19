#!/usr/bin/env python/Users/sameera/Desktop/setup.py

import setuptools
from distutils.core import setup, Extension
from setupext import ext_modules
import numpy as np

setup(
    name='NSLS2',
    version='0',
    author='Brookhaven National Lab',
    packages=["nsls2",
              "nsls2.testing"],
    include_dirs=[np.get_include()],
    ext_modules=ext_modules
    )
