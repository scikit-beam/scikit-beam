=========
 Changes
=========
- Changed function arguments name.
- Each function has only one output.

- from ::

     elastic_peak(x, coherent_sct_energy, fwhm_offset, fwhm_fanoprime, area, epsilon=2.96)

  to ::

     elastic_peak(x, coherent_sct_energy, fwhm_offset, fwhm_fanoprime, coherent_sct_amplitude, epsilon=2.96)

- from ::

     def compton_peak(x, coherent_sct_energy, fwhm_offset, fwhm_fanoprime,
                     compton_angle, compton_fwhm_corr, compton_amplitude,
                     compton_f_step, compton_f_tail, compton_gamma,
                     compton_hi_f_tail, compton_hi_gamma,
                     area, epsilon=2.96, matrix=False)

  to ::

     def compton_peak(x, coherent_sct_energy, fwhm_offset, fwhm_fanoprime,
                      compton_angle, compton_fwhm_corr, compton_amplitude,
                      compton_f_step, compton_f_tail, compton_gamma,
                      compton_hi_f_tail, compton_hi_gamma,
                      epsilon=2.96, matrix=False)
