
# scikit-beam

[![codecov.io](http://codecov.io/github/scikit-beam/scikit-beam/coverage.svg?branch=main)](http://codecov.io/github/scikit-beam/scikit-beam?branch=main)


**[Documentation](http://scikit-beam.github.io/scikit-beam/)**

## Examples
[scikit-beam-examples repository](https://github.com/scikit-beam/scikit-beam-examples)

- [Powder calibration (still needs tilt correction)](https://github.com/scikit-beam/scikit-beam-examples/blob/main/demos/powder_calibration/D_estimate_demo.ipynb)
- 1-time correlation
  - [dir](https://github.com/scikit-beam/scikit-beam-examples/tree/main/demos/1_time_correlation)
  - [Jupyter notebook](https://github.com/scikit-beam/scikit-beam-examples/blob/main/demos/1_time_correlation/Multi_tau_one_time_correlation_example.ipynb)
- Differential Phase Contrast
  - [dir](https://github.com/scikit-beam/scikit-beam-examples/blob/main/demos/dpc)
  - [Jupyter notebook](https://github.com/scikit-beam/scikit-beam-examples/blob/main/demos/dpc/dpc_demo.ipynb)
- [Fast conversion to reciprocal space](https://github.com/scikit-beam/scikit-beam-examples/blob/main/demos/reciprocal_space/recip_example.ipynb)
- [X-Ray Speckle Visibility Spectroscopy](https://github.com/scikit-beam/scikit-beam-examples/blob/main/demos/speckle/speckle-plotting.ipynb)
- [Basic Plotting of X-Ray Fluorescence Elemental Lines](https://github.com/scikit-beam/scikit-beam-examples/blob/main/demos/xrf/plot_xrf_spectrum.ipynb)

## Quick start

### install with conda

```
conda install scikit-beam -c nsls2forge

```

### install development version with setuptools

```
git clone git@github.com:scikit-beam/scikit-beam.git
cd scikit-beam
python setup.py install
```

### set up for development
```
git clone git@github.com:scikit-beam/scikit-beam.git
cd scikit-beam
python setup.py develop
pip install pytest coverage setuptools
```
**make sure all the tests pass!**
```
python run_tests.py
```

**and you can check the code coverage with**
```
coverage run run_tests.py
coverage report -m
```
