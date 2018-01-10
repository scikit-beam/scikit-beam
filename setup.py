#!/usr/bin/env python

import setuptools
from distutils.core import setup, Extension
import versioneer
import os
import sys
import importlib

# Install required dependencies before setup()
deps = {
        'numpy': {
            'module': 'numpy',
            'function': None,
            'alias': 'np'
            },
        'cython': {
            'module': 'Cython.Build',
            'function': 'cythonize',
            'alias': None,
            },
        }


def _import(module, function=None, alias=None):
    if function:
        globals()[function] = getattr(importlib.import_module(module), function)
    if alias:
        globals()[alias] = importlib.import_module(module)


for lib, dep in deps.items():
    try:
        _import(**dep)
    except ImportError:
        import pip
        pip.main(['install', lib])
        _import(**dep)


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def c_ext():
    if os.name == 'nt':
        # we are on windows. Do not compile the extension. Tons of errors are
        # spit out when we compile on AppVeyor.
        # https://gist.github.com/ericdill/bdc86eb81e338ca4624b
        return []

    # compile for MacOS without openmp
    if sys.platform == 'darwin':
        return [Extension('skbeam.ext.ctrans', ['src/ctrans.c'])]
    # compile the extension on Linux.
    return [Extension('skbeam.ext.ctrans', ['src/ctrans.c'],
                      extra_compile_args=['-fopenmp'],
                      extra_link_args=['-lgomp'])]


def cython_ext():
    return cythonize("**/*.pyx")


setup(
    name='scikit-beam',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='Brookhaven National Lab',
    description="Data analysis tools for X-ray science",
    packages=setuptools.find_packages(exclude=['doc']),
    include_dirs=[np.get_include()],
    package_data={'skbeam.core.constants': ['data/*.dat']},
    install_requires=['six', 'numpy'],  # essential deps only
    ext_modules=c_ext() + cython_ext(),
    url='http://github.com/scikit-beam/scikit-beam',
    keywords='Xray Analysis',
    license='BSD',
    classifiers=['Development Status :: 3 - Alpha',
                 "License :: OSI Approved :: BSD License",
                 "Programming Language :: Python :: 2.7",
                 "Programming Language :: Python :: 3.4",
                 "Topic :: Scientific/Engineering :: Physics",
                 "Topic :: Scientific/Engineering :: Chemistry",
                 "Topic :: Software Development :: Libraries",
                 "Intended Audience :: Science/Research",
                 "Intended Audience :: Developers",
                 ],
    )
