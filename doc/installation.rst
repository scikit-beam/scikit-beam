.. _installation:

Installing scikit-xray 
----------------------

For Python Novices
^^^^^^^^^^^^^^^^^^

Installation is simple on Windows, OSX, and Linux, even for Python novices.

1. Get Scientific Python
""""""""""""""""""""""""

To get started with Python on any platform, download and install
`Anaconda <https://store.continuum.io/cshop/anaconda/>`_. It comes with the
common scientific Python packages built in.

2. Install scikit-xray 
""""""""""""""""""""""

Open a command prompt. On Windows, you can use the "Anaconda Command Prompt"
installed by Anaconda or Start > Applications > Command Prompt. On a Mac, look
for Applications > Utilities > Terminal. Type these commands:

.. code-block:: bash

   conda update conda
   conda install -c tacaswell scikit-xray

The above installs scikit-xray and all its requirements. Our tutorials also use
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

We strongly recommend using conda install scikit-xray, as described above,
but pip is also supported.

Essential Dependencies:

* python (both 2 and 3 are supported)
* setuptools
* numpy
* scipy
* six
* xraylib
* scikit-image
* lmfit
* netcdf4

.. code-block:: bash

   git clone https://github.com/Nikea/scikit-xray
   pip install -e scikit-xray

Updating Your Installation
--------------------------

The code is under active development. To update to the latest stable release,
run this in the command prompt:

.. code-block:: bash

    conda update -c tacaswell scikit-xray
