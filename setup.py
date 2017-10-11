#!/usr/bin/env python

import setuptools
import versioneer

from skbuild import setup

setup(
    name='scikit-beam',

    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),

    author='Brookhaven National Lab',

    description="Data analysis tools for X-ray science",

    packages=setuptools.find_packages(exclude=['doc']),
    package_data={
        'skbeam.core.constants': ['data/*.dat']
    },

    install_requires=[
        'six',
        'numpy'
    ],  # essential dependencies only

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
