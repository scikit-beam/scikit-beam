.. _installation:

Installing scikit-beam
----------------------

For Python Novices
^^^^^^^^^^^^^^^^^^

Installation is simple on Windows, OSX, and Linux, even for Python novices.

1. Get Scientific Python
""""""""""""""""""""""""

To get started with Python on any platform, download and install
`Anaconda <https://store.continuum.io/cshop/anaconda/>`_. It comes with the
common scientific Python packages built in.

2. Install scikit-beam
""""""""""""""""""""""

TODO: make this actually work!

Open a command prompt. On Windows, you can use the "Anaconda Command Prompt"
installed by Anaconda or Start > Applications > Command Prompt. On a Mac, look
for Applications > Utilities > Terminal. Type these commands:

.. code-block:: bash

   conda update conda
   conda config --add channels scikit-beam
   # to install the latest stable release
   conda install scikit-beam

The above installs scikit-beam and all its requirements. Our tutorials also use
the IPython notebook. To install that as well, type

.. code-block:: bash

    conda install ipython-notebook

3. Try it out!
""""""""""""""

Finally, to try it out, type

.. code-block:: bash

    ipython notebook

This will automatically open a browser tab, ready to interpret Python code.
To get started, check out the links to tutorials at the top of this document.

More Information for Experienced Python Users
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We strongly recommend using conda install scikit-beam, as described above,
but pip is also supported.

Essential Dependencies:

* python (both 2 and 3 are supported)
* setuptools
* six
* numpy

Optional Dependencies:

* scipy
* scikit-image
* xraylib
* lmfit
* netcdf4

.. code-block:: bash

   git clone https://github.com/scikit-beam/scikit-beam
   pip install -e scikit-beam

Updating Your Installation
--------------------------

The code is under active development. To update to the latest **stable** release,
run this in the command prompt:

.. code-block:: bash

    conda update -c scikit-beam scikit-beam


The code is under active development. To update to the latest **development**
release, run this in the command prompt:

.. code-block:: bash

    conda update -c scikit-beam/channels/dev scikit-beam
