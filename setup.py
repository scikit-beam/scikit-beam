#!/usr/bin/env python

import setuptools
from distutils.core import setup, Extension
from setupext import ext_modules
import numpy as np

setup(
    name='skxray',
    version='0.0.x',
    author='Brookhaven National Lab',
    packages=["skxray",
              "skxray.testing",
              "skxray.fitting",
              "skxray.fitting.model",
              "skxray.fitting.base",
              "skxray.io",
              ],
    include_dirs=[np.get_include()],
    ext_modules=ext_modules
    )
