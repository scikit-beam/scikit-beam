.. _introduction:

Introduction to scikit-beam
---------------------------

scikit-beam provides simple functions useful for the X-ray, Neutron and
Electron communities.  The primary goal of the scikit-beam project is to
provide a centralized repository for algorithms that are used in these three
scientific domains.  The algorithms that we provide within scikit-beam are
fully tested and are usually validated by the scientist who developed the
original technique, so users can have confidence that these algorithms are
accurate and will continue to be accurate.

There are several ways to use scikit-beam. Users comfortable with Python,
IPython, or the Jupyter notebook can use it like any other package. Users
who prefer drag-and-drop software can access all the tools in scikit-beam
through `vistrails <http://www.vistrails.org/index.php/Main_Page>`__.

scikit-beam functions accept and return standard Python and numpy datatypes, so
they play nicely with other packages from the scientific Python community.
Further, the modular design of scikit-beam allows its components to be easily
reused in way not envisioned by the authors.

Scikit-beam is being developed at the National Synchrotron Light Source at
Brookhaven National Lab and also in collaboration with scientists at the LCLS,
APS and the SwissFEL

Targeted Techniques
^^^^^^^^^^^^^^^^^^^
The following is a list of algorithms that are currently available in
scikit-beam along with some planned work for the future.  If any of the planned
work looks interesting to you, please jump in and contribute on `github
<https://github.com/scikit-beam/scikit-beam>`_!

See our :doc:`/example` section for curated Jupyter notebooks that walk through
using these algorithms.

Currently implemented
=====================

* Differential Phase Contrast
* MultiTau correlation

    * 1-time
    * 2-time
    * 1-time from 2-time

* Fast 2-D image conversion to Q
* Fast gridding of 3-D point cloud into 2-D plane

Under active development
========================

* Powder Diffraction
* Image Segmentation
* Tomography

    * Absorption
    * Fluorescence

* Correlation

    * 4-time

Planned
=======

* Ptychography
* Inelastic Scattering
* Coherent Diffractive Imaging
* GPU implementation of Multi-tau correlation
* XANES (1-D, 2-D)

Credit
^^^^^^

scikit-beam is part of the `scikit-beam <https://github.com/scikit-beam>`__
software collaboration supported by `Brookhaven National Lab <http://www.bnl.gov>`__.
