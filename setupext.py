#!/usr/bin/env python
# Copyright (c) Brookhaven National Lab 2O14
# All rights reserved
# BSD License
# See LICENSE for full text

"""
     setupext.py is for c routines

"""

import os
import sys
import six
from six.moves import configparser
import numpy as np
import copy
from distutils.core import setup, Extension


options = {'build_ctrans': False}


ext_default = {'include_dirs': [np.get_include()],
               'library_dirs': [],
               'libraries': [],
               'define_macros': []}


setup_files = ['setup.cfg.%s' % sys.platform, 'setup.cfg']


def detectCPUs():
    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if "SC_NPROCESSORS_ONLN" in os.sysconf_names:
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
            else:
                # OSX:
                return int(os.popen2("sysctl -n hw.ncpu")[1].read())
    # Windows:
    if os.environ.has_key("NUMBER_OF_PROCESSORS"):
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"]);
    if ncpus > 0:
        return ncpus
    return 1  # Default


def parseExtensionSetup(name, config, default):
    default = copy.deepcopy(default)

    try:
        default['include_dirs'] = config.get(name, "include_dirs").split(os.pathsep)
    except:
        pass

    try:
        default['library_dirs'] = config.get(name, "library_dirs").split(os.pathsep)
    except:
        pass

    try:
        default['libraries'] = config.get(name, "libraries").split(",")
    except:
        pass

    return default


setupfile = None


for f in setup_files:
    if os.path.exists(f):
        setupfile = f
        break


if setupfile is not None:
    config = configparser.SafeConfigParser()
    config.read(setupfile)
    print(config)
    try:
        options['build_ctrans'] = config.getboolean("ctrans", "build")
    except:
        pass

    ctrans = parseExtensionSetup('ctrans', config, ext_default)
    threads = False
    try:
        threads = config.getboolean("ctrans", "usethreads")
    except:
        pass

    nthreads = detectCPUs() * 2
    try:
        nthreads = config.getint("ctrans", "max_threads")
    except:
        pass

    if threads:
        ctrans['define_macros'].append(('USE_THREADS', None))
        ctrans['define_macros'].append(('NTHREADS', nthreads))


ext_modules = []
if options['build_ctrans']:
    ext_modules.append(Extension('ctrans', ['src/ctrans.c'],
                                 **ctrans))
