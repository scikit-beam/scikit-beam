#!/usr/bin/env python

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
    )
