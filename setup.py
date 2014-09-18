#!/usr/bin/env python

import setuptools
from distutils.core import setup, Extension
from setupext import ext_modules
import numpy as np

setup(
    name='NSLS2',
    version='0.0.x',
    author='Brookhaven National Lab',
    packages=["nsls2",
              "nsls2.testing",
              "nsls2.fitting",
              "nsls2.fitting.model",
              "nsls2.fitting.base",
              "nsls2.io",
              "nsls2.img_proc",
    ],
    include_dirs=[np.get_include()],
    ext_modules=ext_modules, requires=[]
)
