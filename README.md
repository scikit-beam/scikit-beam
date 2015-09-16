[![Build Status](https://travis-ci.org/scikit-xray/scikit-xray.svg?branch=master)](https://travis-ci.org/scikit-xray/scikit-xray)
[![codecov.io](http://codecov.io/github/scikit-xray/scikit-xray/coverage.svg?branch=master)](http://codecov.io/github/scikit-xray/scikit-xray?branch=master)
[![Coverage Status](https://coveralls.io/repos/scikit-xray/scikit-xray/badge.svg?branch=master&service=github)](https://coveralls.io/github/scikit-xray/scikit-xray?branch=master)

# scikit-xray

**[Documentation] (http://scikit-xray.github.io/scikit-xray/)**

## Examples
[scikit-xray-examples repository] (https://github.com/scikit-xray/scikit-xray-examples)

- [Powder calibration (still needs tilt correction)] (https://github.com/scikit-xray/scikit-xray-examples/blob/master/notebooks/2014-09-15_calibration/D_estimate_demo.ipynb>)
- [1-time correlation] (https://github.com/scikit-xray/scikit-xray-examples/tree/master/demos/1_time_correlation)
  [Jupyter notebook] (https://github.com/scikit-xray/scikit-xray-examples/blob/master/demos/1_time_correlation/Multi_tau_one_time_correlation_example.ipynb)
- [Differential Phase Contrast] (https://github.com/scikit-xray/scikit-xray-examples/blob/master/demos/dpc)
  [Jupyter notebook] (https://github.com/scikit-xray/scikit-xray-examples/blob/master/demos/dpc/dpc_demo.ipynb)
- [Fast conversion to reciprocal space] (https://github.com/scikit-xray/scikit-xray-examples/blob/master/demos/reciprocal_space/recip_example.ipynb)
- [X-Ray Speckle Visibility Spectroscopy] (https://github.com/scikit-xray/scikit-xray-examples/blob/master/demos/speckle/speckle-plotting.ipynb)
- [Basic Plotting of X-Ray Fluorescence Elemental Lines] (https://github.com/scikit-xray/scikit-xray-examples/blob/master/demos/xrf/plot_xrf_spectrum.ipynb)

## Quick start

### install with conda

```
conda config --add channels scikit-xray
conda install scikit-xray
```

### install with pip

```
git clone git@github.com:scikit-xray/scikit-xray.git
cd scikit-xray
python setup.py install
```

### set up for development
```
git clone git@github.com:scikit-xray/scikit-xray.git
cd scikit-xray
python setup.py develop
pip install nose coverage setuptools
```
**make sure all the tests pass!**
```
python run_tests.py
```
