======================
 :mod:`core` subpackge
======================

.. currentmodule:: skxray.core


Scientific Algorithms
---------------------

.. autosummary::
   :toctree: generated/
   
   cdi.cdi_recon
   correlation.multi_tau_auto_corr
   dpc.recon
   dpc.dpc_runner
   recip.process_to_q


Helper Classes
--------------

Dictionary-like classes
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   utils.MD_dict
   utils.verbosedict
   utils.RCParamDict

Image warping functions
-----------------------
.. autosummary::
   :toctree: generated/

   utils.img_to_relative_xyi
   utils.radial_grid
   utils.angle_grid


Peak
----

Peak fitting
~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/
   
   feature.peak_refinement
   feature.refine_quadratic
   feature.refine_log_quadratic
   feature.filter_n_largest
   feature.filter_peak_height

Peak finding
~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/
   
   image.find_ring_center_acorr_1D
   spectroscopy.find_largest_peak

Image pre-processing
--------------------
.. autosummary::
   :toctree: generated/

   utils.subtract_reference_images


Histograms and Integration
--------------------------

Binning
~~~~~~~
.. autosummary::
   :toctree: generated/

   utils.bin_1D
   utils.wedge_integration
   utils.grid3d


Helper functions
----------------
.. autosummary::
   :toctree: generated/

   utils.pairwise
   utils.geometric_series
   utils.multi_tau_lags
   utils.bin_edges
   utils.bin_edges_to_centers

Generating ROIs
---------------
.. autosummary::
   :toctree: generated/

   roi.roi_kymograph
   roi.circular_average
   roi.combine_mean_intensity
   roi.mean_intensity
   roi.mean_intensity_sets
   roi.roi_pixel_values
   roi.roi_max_counts
   roi.segmented_rings
   roi.ring_edges
   roi.rings
   roi.rectangles
   
   
Physical relations
------------------
.. autosummary::
   :toctree: generated/

   utils.q_to_d
   utils.d_to_q
   utils.q_to_twotheta
   utils.twotheta_to_q
   utils.radius_to_twotheta
   recip.hkl_to_q
   recip.calibrated_pixels_to_q


Boolean Logic
-------------
.. autosummary::
   :toctree: generated/
   
    arithmetic.logical_nand
    arithmetic.logical_nor
    arithmetic.logical_sub

Calibration
-----------
.. autosummary::
   :toctree: generated/
   
   utils.estimate_d_blind
   utils.refine_center
   
