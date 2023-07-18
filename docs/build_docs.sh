#!/bin/bash
set -e # exit with nonzero exit code if anything fails

make clean
# The original Makefile was doing nothing if the option 'notebooks' was selected.
# So this functionality was not supported (or broken)
# make notebooks
make html
