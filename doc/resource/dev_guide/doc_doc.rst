.. _doc_doc:

Documentation
=============

This is documentation on how to build and add to our documentation.

Fully documenting the library is of utmost important.  It is more
valuable to have fully documented library with fewer features than
a feature-rich library no one can figure out how to use.

Docstrings
----------

The docstrings must follow the `numpydoc
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
format.

Sphinx
------

We are using `sphinx <http://sphinx-doc.org/>`_ to build the
documentation. In addition to `sphinx` you will also need `numpydoc
<https://pypi.python.org/pypi/numpydoc>`_ installed and available.  Both
can be installed from pypi (:code:`pip install numpydoc` and
:code:`pip install sphinx`).  If you want to build a pdf version of the
documentation you will also need LaTeX.

To build the documentation locally, navigate to the `doc` folder and run ::

    make html

The output website will then be in `_build/html/index.html` which you can
open using any web-browser.
