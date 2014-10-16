====================
 :mod:`core` Module
====================

.. currentmodule:: nsls2.core


Helper Classes
--------------

Dictionary-like classes
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   MD_dict
   verbosedict
   RCParamDict

Image warping functions
-----------------------


.. autosummary::
   :toctree: generated/

   detector2D_to_1D
   pixel_to_radius
   pixel_to_phi

Image pre-processing
--------------------

.. autosummary::
   :toctree: generated/

   img_subtraction_pre

Histograms and Integration
--------------------------

Binning
~~~~~~~

.. autosummary::
   :toctree: generated/

   bin_1D
   bin_image_to_1D
   wedge_integration
   grid3d


Utility functions
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   bin_edges
   bin_edges_to_centers


Helper functions
----------------
.. autosummary::
   :toctree: generated/

   pairwise

Physical relations
------------------
.. autosummary::
   :toctree: generated/

   q_to_d
   d_to_q
   q_to_twotheta
   twotheta_to_q
