.. raw:: html

    <style type="text/css">
    .thumbnail {{
        position: relative;
        float: left;
        margin: 10px;
        width: 180px;
        height: 200px;
    }}

    .thumbnail img {{
        position: absolute;
        display: inline;
        left: 0;
        width: 170px;
        height: 170px;
    }}

    </style>


scikit-xray: Data Analysis Tools for X-ray Science
==================================================

.. raw:: html

    <div style="clear: both"></div>
    <div class="container-fluid hidden-xs hidden-sm">
      <div class="row">
        <div class="col-md-2 thumbnail">
        </div>
        <div class="col-md-2 thumbnail">
        </div>
        <div class="col-md-2 thumbnail">
        </div>
        <div class="col-md-2 thumbnail">
        </div>
        <div class="col-md-2 thumbnail">
        </div>
        <div class="col-md-2 thumbnail">
        </div>
      </div>
    </div>
    <br>


Scikit-xray is a Python package providing tools for X-ray science.
For a brief introduction to the ideas behind the package, you can read the
:ref:`introductory notes <introduction>`.

Users who prefer drag-and-drop software may prefer to use
`vistrails <http://www.vistrails.org/index.php/Main_Page>`__, through which
all the functionality of scikit-xray is also available.
(Think LabView for X-ray image processing and data analysis.)


The :ref:`tutorial <tutorial>` walks through several experiments from start to
finish -- from the raw data to a publication-style plot. You can also browse
the :ref:`API reference <api_ref>` for a quick overview of all the available
tools.

To check out the code, report a bug, or contribute a new feature, please visit
the `github repository <https://github.com/scikit-xray/scikit-xray>`_.

.. raw:: html

   <div class="container-fluid">
   <div class="row">
   <div class="col-md-6">
   <h2>Documentation</h2>

.. toctree::
   :maxdepth: 1

   introduction
   installation
   whats_new
   resource/api/index
   resource/dev_guide/index
   examples
   tutorial

.. raw:: html

   </div>
   <div class="col-md-6">
   <h2>Tutorial</h2>

.. toctree::
   :maxdepth: 1

   tutorial/dpc_demo
   tutorial/speckle-plotting
   tutorial/XPCS_fitting_with_lmfit
   tutorial/Multi_tau_one_time_correlation_example
   tutorial/Generate_ROI_labeled_arrays
   tutorial/recip_example
   tutorial/plot_xrf_spectrum

.. raw:: html

   </div>
   </div>
   </div>
