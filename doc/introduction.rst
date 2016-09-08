.. _introduction:

Introduction to scikit-beam
---------------------------

Targeted Techniques
^^^^^^^^^^^^^^^^^^^
The following is a list of algorithms that are currently available in
scikit-beam along with some planned work for the future.  If any of the planned
work looks interesting to you, please jump in and contribute on `github
<https://github.com/scikit-beam/scikit-beam>`_!

.. See our :doc:`/example` section for curated Jupyter notebooks that walk through
   using these algorithms.

Currently implemented
=====================

* Differential Phase Contrast (:mod:`~skbeam.core.dpc`)
* CDI (:mod:`~skbeam.core.cdi`)
* MultiTau correlation
* Fast 2-D image conversion to Q
* Fast gridding of 3-D point cloud into 2-D plane
* X-Ray Speckle Visibility Spectroscopy (XSVS)
* X-ray Fluorescence
    * `Fitting GUI <https://github.com/NSLS-II/pyxrf>`_
* Fast histograms
* Access to basic constants (:mod:`~skbeam.core.constants`)


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
