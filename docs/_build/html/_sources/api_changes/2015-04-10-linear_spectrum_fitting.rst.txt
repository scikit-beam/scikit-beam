Change return tuple on linear_spectrum_fitting function
-------------------------------------------------------

The return value of linear_spectrum_fitting function in the
``fitting/xrf_model.py`` module was changed to include the peak area of each
element.  This function now returns three arguments instead of two.
