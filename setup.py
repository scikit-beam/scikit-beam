#!/usr/bin/env python

import setuptools
from distutils.core import setup, Extension
from setupext import ext_modules

setup(
    name='NSLS2',
    version='0',
    author='Brookhaven National Lab',
    packages=["nsls2"],
    include_dirs=[np.get_include()],
    ext_modules=[ext_modules]
    )
