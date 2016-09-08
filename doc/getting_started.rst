==================================
 Getting Started with Scikit-beam
==================================

Importing scikit-beam
=====================

In order to encourage consistency amongst users in importing and using Scikit-beam
functionality, we have put together the following guidelines.

Since most of the functionality in Scikit-beam resides in sub-packages, importing
scikit-beam as::

    >>> import skbeam

is not very useful. Instead, it is best to import the desired sub-package
with the syntax::

    >>> from skbeam import subpackage  # doctest: +SKIP

For example, to access the correlation-related functionality, you can import
`skbeam.core.correlation` with::

    >>> from skbeam.core import correlation as corr
    >>> g2 = corr.multi_tau_auto_corr(5, 3, labels, img_seq)


Note that for clarity, and to avoid any issues, we recommend to **never**
import any skbeam functionality using ``*``, for example::

    >>> from skbeam.core.correlation import *  # NOT recommended

Some components of Scikit-Beam started off as standalone packages
(e.g. PyFITS, PyWCS), so in cases where Scikit-Beam needs to be used
as a drop-in replacement, the following syntax is also acceptable::

    >>> from skbeam.io import foo as bar

Getting started with subpackages
================================

.. warning::

   This is not implemented in skbeam yet

Because different subpackages have very different functionality, further
suggestions for getting started are in the documentation for the subpackages,
which you can reach by browsing the sections listed in the :ref:`user-docs`.

Or, if you want to dive right in, you can either look at docstrings for
particular a package or object, or access their documentation using the
:func:`~skbeam.find_api_page` function. For example, doing this::

    >>> from skbeam import find_api_page
    >>> find_api_page(corr.multi_tau_auto_corr)  # doctest: +SKIP

Will bring up the documentation for the
:func:`~skbeam.core.correlation.multi_tau_auto_corr` in your browser.
