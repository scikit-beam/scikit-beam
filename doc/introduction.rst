.. _introduction:

Introduction to scikit-xray
---------------------------

Scikit-xray provides simple functions useful for X-ray science, conveniently
grouped by technique. It includes domain-specific functionality written for
this package and also relevant functions from other scientific packages such as
scikit-image. Thus, it is a complete solution, curating useful tools from the
scientific Python community and presenting them in a context for X-ray science.

There are several ways to use scikit-xray. Users comfortable with Python,
IPython, or the IPython notebook can use it like any other package. Users
who prefer drag-and-drop software can access all the tools in scikit-xray
through `vistrails <http://www.vistrails.org/index.php/Main_Page>`__.

Scikit-xray functions accept and return standard Python and numpy datatypes, so
they play nicely with other packages from the scientific Python community.
Further, the modular design of scikit-xray allows its components to be easily
reused in way not envisioned by the authors.

Targeted Techniques
^^^^^^^^^^^^^^^^^^^
Scikit-xray is being developed to support X-ray techniques at the beamlines
listed in the Supported Beamlines section.

Currently implemented
=====================

* Differential Phase Contrast

Under active development
========================

* Powder Diffraction
* X-ray Fluorescence
* Image Segmentation
* Tomography

   * Absorption
   * Fluorescence

* 1-time correlation

Planned
=======

* Ptychography
* Inelastic Scattering
* Coherent Diffractive Imaging
* 2-time correlation
* XANES (1-D, 2-D)

Supported Beamlines
^^^^^^^^^^^^^^^^^^^
Scikit-image is developed in collaboration with beamline scientists at
the following beamlines.

**NSLS-II**

* Inelastic X-ray Scattering (IXS)
* X-ray Powder Diffraction (XPD)
* Coherent Hard X-ray Scattering (CHX)
* Coherent Soft X-ray Scattering (CSX)
* Submicron Resolution X-ray Spectroscopy (SRX)
* Hard X-ray Nanoprobe (HXN)

Credit
^^^^^^

Scikit-xray is part of the `Nikea <https://github.com/Nikea>`__ software
organization supported by the 
`Brookhaven National Lab <http://www.bnl.gov>`__.
